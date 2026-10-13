"""Deterministic ODE simulation of TemplateModels for the MCP server.

MIRA's ``OdeModel`` builds the right-hand side; integration uses
``scipy.integrate.solve_ivp`` so solver failures, time-dependent rate laws
and runaway stiff systems are reported instead of silently returning
garbage.

Values are resolved in increasing order of precedence: values stored in the
model, a scenario from the database, then explicit overrides. Scenarios
usually carry initial conditions as parameters named after the compartment
(``E_0``, ``I0``, ...), and these are mapped onto the model's initials.
"""

import csv
import io
import logging
import math
import threading
import time
import uuid
from collections import OrderedDict

import numpy as np
import scipy.integrate
import sympy

from mira.metamodel import TemplateModel
from mira.metamodel.template_model import Initial
from mira.modeling import Model
from mira.modeling.ode import OdeModel

from miradb.mcp.mira_tools import (
    ensure_time,
    parse_expression,
    undeclared_rate_symbols,
)
from miradb.mcp.model_store import ModelStore

logger = logging.getLogger(__name__)

MAX_POINTS = 5000
DEFAULT_TIMEOUT_S = 30.0
PREVIEW_POINTS = 11
SOLVER_METHODS = ("LSODA", "RK45", "BDF", "Radau", "DOP853")
INITIAL_SUFFIXES = ("_0", "0", "(0)", "_init", "_initial")


class SimulationTimeout(RuntimeError):
    pass


class SimulationStore:
    """Hold simulation results so the agent can fetch more detail later."""

    def __init__(self, max_results: int = 100):
        self._max = max_results
        self._results: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, result: dict) -> str:
        sim_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._results[sim_id] = result
            while len(self._results) > self._max:
                self._results.popitem(last=False)
        return sim_id

    def get(self, sim_id: str) -> dict:
        with self._lock:
            if sim_id not in self._results:
                raise LookupError(
                    f"No simulation {sim_id!r}. Results do not survive a "
                    "server restart."
                )
            return self._results[sim_id]


def _symbol_keys(namespace: dict[str, float]) -> dict:
    # String keys would be sympified, turning e.g. "N" into sympy.N
    return {sympy.Symbol(k): v for k, v in namespace.items()}


def _scenario_value(param: dict, namespace: dict) -> float | None:
    if param.get("value") is not None:
        value = float(param["value"])
        return None if math.isnan(value) else value
    expression = param.get("value_expression")
    if not expression:
        return None
    try:
        value = parse_expression(expression).subs(_symbol_keys(namespace))
        return float(value)
    except (sympy.SympifyError, TypeError, ValueError, SyntaxError):
        return None


def _initial_target(name: str, compartments: set[str]) -> str | None:
    """Return the compartment a scenario parameter like ``I_0`` sets."""
    for suffix in INITIAL_SUFFIXES:
        if name.endswith(suffix) and name[: -len(suffix)] in compartments:
            return name[: -len(suffix)]
    lowered = {c.lower(): c for c in compartments}
    for suffix in INITIAL_SUFFIXES:
        if name.lower().endswith(suffix):
            base = name[: -len(suffix)].lower()
            if base in lowered:
                return lowered[base]
    return None


def apply_scenario(tm: TemplateModel, scenario: dict) -> tuple[dict, dict]:
    """Apply a scenario's parameters (from ``db_tools.get_scenario``) to a
    model in place.

    Returns
    -------
    report :
        Keys ``parameters`` and ``initials`` (what was set), ``unused``
        (scenario parameters that matched nothing) and ``no_value``
        (scenario parameters without a numeric value).
    namespace :
        Every resolved scenario value by name, so initial expressions such
        as ``N - I0`` can reference them even when ``I0`` itself was mapped
        onto an initial.
    """
    compartments = set(tm.get_concepts_name_map())
    undeclared = undeclared_rate_symbols(tm)
    namespace: dict[str, float] = {}
    params = scenario.get("parameters") or []
    # Two passes so value expressions can reference other scenario values
    for _ in range(2):
        for param in params:
            value = _scenario_value(param, namespace)
            if value is not None:
                namespace[param["name"]] = value

    report = {"parameters": {}, "initials": {}, "unused": {}, "no_value": []}
    concepts = tm.get_concepts_name_map()
    for param in params:
        name = param["name"]
        value = namespace.get(name)
        if value is None:
            report["no_value"].append(name)
            continue
        if name in tm.parameters or name in undeclared:
            tm.set_parameters({name: value})
            report["parameters"][name] = value
        elif target := _initial_target(name, compartments):
            tm.initials[target] = Initial(
                concept=concepts[target], expression=sympy.Float(value)
            )
            report["initials"][target] = value
        else:
            report["unused"][name] = value
    return report, namespace


def _column_stats(times: np.ndarray, series: np.ndarray) -> dict:
    imax = int(np.argmax(series))
    return {
        "initial": float(series[0]),
        "final": float(series[-1]),
        "min": float(series.min()),
        "max": float(series[imax]),
        "t_at_max": float(times[imax]),
    }


def _downsample_indices(n: int, k: int) -> np.ndarray:
    if n <= k:
        return np.arange(n)
    return np.unique(np.linspace(0, n - 1, k).round().astype(int))


def simulate(
    store: ModelStore,
    sims: SimulationStore,
    model_ref: str,
    t_end: float,
    *,
    t_start: float = 0.0,
    n_points: int = 101,
    parameters: dict[str, float] | None = None,
    initials: dict[str, float | str] | None = None,
    scenario: dict | None = None,
    method: str = "LSODA",
    timeout_s: float = DEFAULT_TIMEOUT_S,
) -> dict:
    """Simulate a model and return a summary of the trajectories.

    Returns a dict whose ``status`` is ``ok``, ``missing_values`` (nothing
    was simulated; the dict says which values to supply) or
    ``solver_failed``.
    """
    if t_end <= t_start:
        raise ValueError("t_end must be greater than t_start.")
    if method not in SOLVER_METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {SOLVER_METHODS}.")
    n_points = max(2, min(int(n_points), MAX_POINTS))
    ref = store.normalize_ref(model_ref)
    tm = ensure_time(store.get(ref))

    scenario_report, scenario_namespace = (
        apply_scenario(tm, scenario) if scenario else (None, {})
    )
    undeclared = undeclared_rate_symbols(tm)
    for name, value in (parameters or {}).items():
        if name in tm.parameters or name in undeclared:
            tm.set_parameters({name: float(value)})
        else:
            raise ValueError(
                f"Unknown parameter {name!r}. Known: {sorted(tm.parameters)}."
            )
    concepts = tm.get_concepts_name_map()
    for name, value in (initials or {}).items():
        if name not in concepts:
            raise ValueError(
                f"Unknown compartment {name!r}. Known: {sorted(concepts)}."
            )
        tm.initials[name] = Initial(
            concept=concepts[name], expression=parse_expression(value)
        )

    undeclared = undeclared_rate_symbols(tm)
    if undeclared:
        return {
            "status": "missing_values",
            "model_ref": ref,
            "undeclared_rate_symbols": sorted(undeclared),
            "hint": "Rate laws use symbols the model does not declare. "
                    "Pass them in `parameters` to add them.",
            "scenario_applied": scenario_report,
        }

    try:
        om = OdeModel(Model(tm), initialized=False)
    except KeyError as err:
        raise ValueError(
            f"Cannot build ODEs: transition without a usable rate law ({err})."
        ) from err

    param_values: dict[str, float] = {}
    missing_params = []
    for key in om.pmap:
        value = om.model.parameters[key].value
        if value is None:
            missing_params.append(key)
        else:
            param_values[key] = float(value)

    # Initials may reference parameters no rate law uses, e.g. N in S0 = N - I0
    namespace = {
        **scenario_namespace,
        **{n: float(p.value) for n, p in tm.parameters.items()
           if p.value is not None},
        **param_values,
    }
    symbol_namespace = _symbol_keys(namespace)
    y0 = np.zeros(len(om.vmap))
    missing_initials = []
    for key, idx in om.vmap.items():
        name = om.vname_map[idx]
        initial = tm.initials.get(name)
        if initial is None:
            missing_initials.append(name)
            continue
        expr = sympy.sympify(initial.expression).subs(symbol_namespace)
        if expr.free_symbols:
            missing_initials.append(
                f"{name} (expression {initial.expression} needs "
                f"{sorted(map(str, expr.free_symbols))})"
            )
            continue
        y0[idx] = float(expr)

    if missing_params or missing_initials:
        return {
            "status": "missing_values",
            "model_ref": ref,
            "missing_parameters": sorted(missing_params),
            "missing_initials": missing_initials,
            "hint": "Pass the missing values in `parameters` / `initials`, "
                    "or pick a `scenario_id` that provides them.",
            "scenario_applied": scenario_report,
        }

    t_sym = sympy.Symbol("t")
    p_subs = {om.p[idx]: param_values[key] for key, idx in om.pmap.items()}
    kinetics = om.kinetics.xreplace(p_subs)
    observables = om.observables.xreplace(p_subs)
    rhs_fn = sympy.lambdify([t_sym, om.y], kinetics, modules="numpy")
    deadline = time.monotonic() + timeout_s

    def rhs(t, y):
        if time.monotonic() > deadline:
            raise SimulationTimeout(f"Simulation exceeded {timeout_s} s.")
        return np.asarray(rhs_fn(t, y.reshape(-1, 1)), dtype=float).reshape(-1)

    times = np.linspace(t_start, t_end, n_points)
    try:
        sol = scipy.integrate.solve_ivp(
            rhs, (t_start, t_end), y0, t_eval=times, method=method,
            rtol=1e-6, atol=1e-6,
        )
    except SimulationTimeout as err:
        return {"status": "solver_failed", "model_ref": ref, "message": str(err),
                "hint": "Try a shorter t_end or method='BDF' for stiff models."}

    if not sol.success or sol.y.shape[1] != n_points:
        return {"status": "solver_failed", "model_ref": ref,
                "message": sol.message,
                "hint": "Try method='BDF' or 'Radau' for stiff models, or "
                        "check parameter magnitudes."}

    columns = [om.vname_map[i] for i in range(len(om.vmap))]
    values = sol.y.T
    if observables.shape[0]:
        obs_fn = sympy.lambdify([t_sym, om.y], observables, modules="numpy")
        obs_values = np.array([
            np.asarray(obs_fn(t, row.reshape(-1, 1)), dtype=float).reshape(-1)
            for t, row in zip(sol.t, values)
        ])
        columns += list(om.observable_map)
        values = np.hstack([values, obs_values])

    sim_id = sims.add({
        "model_ref": ref,
        "times": sol.t,
        "columns": columns,
        "values": values,
        "n_variables": len(om.vmap),
    })

    n_vars = len(om.vmap)
    total = values[:, :n_vars].sum(axis=1)
    scale = max(float(np.abs(total).max()), 1e-12)
    warnings = []
    negatives = [c for i, c in enumerate(columns[:n_vars])
                 if values[:, i].min() < -1e-6 * scale]
    if negatives:
        warnings.append(f"Negative values in {negatives}; check parameters "
                        "and rate laws.")
    preview_idx = _downsample_indices(len(sol.t), PREVIEW_POINTS)
    return {
        "status": "ok",
        "sim_id": sim_id,
        "model_ref": ref,
        "parameters_used": param_values,
        "initials_used": {c: float(y0[i]) for i, c in enumerate(columns[:n_vars])},
        "scenario_applied": scenario_report,
        "stats": {
            c: _column_stats(sol.t, values[:, i]) for i, c in enumerate(columns)
        },
        "total_of_compartments": {
            "initial": float(total[0]),
            "final": float(total[-1]),
            "relative_change": float((total[-1] - total[0]) / scale),
        },
        "preview": {
            "t": [float(t) for t in sol.t[preview_idx]],
            **{c: [float(v) for v in values[preview_idx, i]]
               for i, c in enumerate(columns)},
        },
        "warnings": warnings,
        "full_results": f"miradb://simulation/{sim_id}.csv",
    }


def get_trajectory(
    sims: SimulationStore,
    sim_id: str,
    variables: list[str] | None = None,
    n_points: int = 50,
) -> dict:
    """Return a downsampled trajectory of selected columns of a simulation."""
    result = sims.get(sim_id)
    columns = result["columns"]
    selected = variables or columns
    unknown = [v for v in selected if v not in columns]
    if unknown:
        raise ValueError(f"Unknown columns {unknown}; available: {columns}.")
    idx = _downsample_indices(len(result["times"]), max(2, int(n_points)))
    return {
        "sim_id": sim_id,
        "model_ref": result["model_ref"],
        "t": [float(t) for t in result["times"][idx]],
        **{
            v: [float(x) for x in result["values"][idx, columns.index(v)]]
            for v in selected
        },
    }


def trajectory_csv(sims: SimulationStore, sim_id: str) -> str:
    """Return the full trajectory of a simulation as CSV."""
    result = sims.get(sim_id)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["t", *result["columns"]])
    for t, row in zip(result["times"], result["values"]):
        writer.writerow([repr(float(t)), *(repr(float(v)) for v in row)])
    return buf.getvalue()
