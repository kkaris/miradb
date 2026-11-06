"""MIRA model operations for the MCP server.

Every operation takes model handles (see ``miradb.mcp.model_store``) and
returns compact summaries; operations that produce a new model store it in
the workspace and return its new handle.
"""

import logging
import re
from collections import Counter

import sympy

from mira.metamodel import TemplateModel
from mira.metamodel.ops import simplify_rate_laws, stratify
from mira.metamodel.template_model import Initial, Time
from mira.modeling import Model
from mira.modeling.amr.petrinet import template_model_to_petrinet_json
from mira.modeling.ode import OdeModel

from miradb.mcp.model_store import ModelStore

logger = logging.getLogger(__name__)

EXPORT_FORMATS = ("template_model", "petrinet_amr")


def _concept_summary(concept) -> dict:
    out = {"name": concept.name}
    if concept.display_name:
        out["display_name"] = concept.display_name
    if concept.identifiers:
        out["identifiers"] = {
            k: v for k, v in concept.identifiers.items()
        }
    if concept.context:
        out["context"] = dict(concept.context)
    return out


def _template_concepts(template):
    return [c for c in template.get_concepts_flat() if c is not None]


def ensure_time(tm: TemplateModel) -> TemplateModel:
    """Give the model a time variable so time-dependent rate laws resolve."""
    if tm.time is None:
        tm.time = Time(name="t", units=None)
    return tm


_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_expression(value) -> sympy.Expr:
    """Parse a number or expression string into a SymPy expression.

    Every identifier is a plain symbol, so names such as ``N``, ``I``,
    ``E``, ``S``, ``beta`` or ``gamma`` are not mistaken for SymPy's
    built-in functions and constants.
    """
    if isinstance(value, (int, float)):
        return sympy.Float(value)
    value = str(value)
    names = {name: sympy.Symbol(name) for name in _IDENTIFIER.findall(value)}
    return sympy.sympify(value, locals=names)


def undeclared_rate_symbols(tm: TemplateModel) -> set[str]:
    """Return symbols used in rate laws that are neither compartments,
    parameters nor the time variable.

    Extracted models occasionally reference a constant (e.g. ``N``) that
    was never declared as a parameter; MIRA cannot build ODEs for them
    until it is.
    """
    known = set(tm.get_concepts_name_map()) | set(tm.parameters or {})
    if tm.time is not None:
        known.add(tm.time.name)
    used = set()
    for template in tm.templates:
        if template.rate_law is not None:
            used |= {str(s) for s in template.rate_law.free_symbols}
    return used - known - {"t"}


def transition_id(template, index: int) -> str:
    """Return the ID tools use for a transition: its name, or ``t<n>``
    (1-based position) for the unnamed transitions of extracted models."""
    return template.name or f"t{index + 1}"


def find_transition(tm: TemplateModel, transition: str) -> int:
    """Return the index of a transition given by name or ``t<n>`` ID."""
    named = [i for i, t in enumerate(tm.templates) if t.name == transition]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        raise ValueError(
            f"{len(named)} transitions are named {transition!r}; refer to "
            f"one by position instead, e.g. 't{named[0] + 1}'."
        )
    match = re.fullmatch(r"t(\d+)", transition)
    if match and 1 <= int(match.group(1)) <= len(tm.templates):
        return int(match.group(1)) - 1
    ids = [transition_id(t, i) for i, t in enumerate(tm.templates)]
    raise ValueError(f"No transition {transition!r}. Transitions: {ids}.")


def describe_template(template, index: int) -> dict:
    return {
        "id": transition_id(template, index),
        "type": template.type,
        "concepts": [c.name for c in _template_concepts(template)],
        "rate_law": str(template.rate_law) if template.rate_law else None,
    }


def summarize_model(tm: TemplateModel, model_ref: str | None = None) -> dict:
    """Return a compact, JSON-serializable description of a TemplateModel."""
    concepts = {}
    transitions = []
    for index, template in enumerate(tm.templates):
        for concept in _template_concepts(template):
            concepts.setdefault(concept.name, _concept_summary(concept))
        transitions.append(
            {
                "id": transition_id(template, index),
                "name": template.name,
                "type": template.type,
                "concepts": [c.name for c in _template_concepts(template)],
                "rate_law": str(template.rate_law) if template.rate_law else None,
            }
        )
    parameters = {
        name: {
            "value": p.value,
            "units": str(p.units.expression) if p.units else None,
            **({"distribution": p.distribution.to_json()}
               if getattr(p, "distribution", None) else {}),
        }
        for name, p in (tm.parameters or {}).items()
    }
    initials = {
        name: str(init.expression) for name, init in (tm.initials or {}).items()
    }
    summary = {
        "compartments": list(concepts.values()),
        "transitions": transitions,
        "transition_type_counts": dict(Counter(t["type"] for t in transitions)),
        "parameters": parameters,
        "parameters_without_value": sorted(
            n for n, p in parameters.items() if p["value"] is None
        ),
        "initials": initials,
        "compartments_without_initial": sorted(
            name for name in concepts if name not in initials
        ),
        "observables": {
            name: str(obs.expression)
            for name, obs in (tm.observables or {}).items()
        },
        "time": tm.time.to_json() if tm.time else None,
        "annotations": tm.annotations.to_json() if tm.annotations else {},
    }
    undeclared = undeclared_rate_symbols(tm)
    if undeclared:
        summary["undeclared_rate_symbols"] = sorted(undeclared)
    if model_ref:
        summary = {"model_ref": model_ref, **summary}
    return summary


def model_equations(tm: TemplateModel) -> list[str]:
    """Return the model's ODEs as plain-text ``d(X)/dt = ...`` strings."""
    om = OdeModel(Model(ensure_time(tm)), initialized=False)
    lines = []
    for lhs, _, rhs in om.get_interpretable_kinetics().tolist():
        var = lhs.args[0] if isinstance(lhs, sympy.Derivative) else lhs
        lines.append(f"d{var}/dt = {rhs}")
    return lines


def get_model(store: ModelStore, model_ref: str) -> dict:
    tm = store.get(model_ref)
    ref = store.normalize_ref(model_ref)
    summary = summarize_model(tm, ref)
    if ref.startswith("ws:"):
        summary["lineage"] = store.lineage(ref)
    return summary


def get_equations(store: ModelStore, model_ref: str) -> dict:
    return {
        "model_ref": store.normalize_ref(model_ref),
        "equations": model_equations(store.get(model_ref)),
    }


def stratify_model(
    store: ModelStore,
    model_ref: str,
    key: str,
    strata: list[str],
    *,
    structure: list[list[str]] | None = None,
    directed: bool = False,
    cartesian_control: bool = False,
    concepts_to_stratify: list[str] | None = None,
    concepts_to_preserve: list[str] | None = None,
    params_to_stratify: list[str] | None = None,
    params_to_preserve: list[str] | None = None,
) -> dict:
    tm = store.get(model_ref)
    new_tm = stratify(
        tm,
        key=key,
        strata=strata,
        structure=[tuple(pair) for pair in structure] if structure else None,
        directed=directed,
        cartesian_control=cartesian_control,
        concepts_to_stratify=concepts_to_stratify,
        concepts_to_preserve=concepts_to_preserve,
        params_to_stratify=params_to_stratify,
        params_to_preserve=params_to_preserve,
    )
    new_ref = store.add(
        new_tm, parent=model_ref, operation=f"stratify(key={key!r}, strata={strata})"
    )
    return summarize_model(new_tm, new_ref)


def simplify_model(store: ModelStore, model_ref: str) -> dict:
    new_tm = simplify_rate_laws(store.get(model_ref))
    new_ref = store.add(new_tm, parent=model_ref, operation="simplify_rate_laws")
    return summarize_model(new_tm, new_ref)


def set_model_values(
    store: ModelStore,
    model_ref: str,
    parameters: dict[str, float] | None = None,
    initials: dict[str, float | str] | None = None,
) -> dict:
    """Return a new workspace model with parameter values and initials set.

    Initials are keyed by compartment name; values may be numbers or
    expressions in the model's parameters (e.g. ``"N - I0"``). A parameter
    that a rate law uses but the model never declared is added.
    """
    tm = store.get(model_ref)
    undeclared = undeclared_rate_symbols(tm)
    unknown_params = []
    for name, value in (parameters or {}).items():
        if name in tm.parameters or name in undeclared:
            tm.set_parameters({name: float(value)})
        else:
            unknown_params.append(name)

    concepts = tm.get_concepts_name_map()
    unknown_initials = []
    for name, value in (initials or {}).items():
        if name not in concepts:
            unknown_initials.append(name)
            continue
        tm.initials[name] = Initial(
            concept=concepts[name], expression=parse_expression(value)
        )

    if unknown_params or unknown_initials:
        raise ValueError(
            f"Unknown parameters {unknown_params} or compartments "
            f"{unknown_initials}. Known parameters: {sorted(tm.parameters)}; "
            f"known compartments: {sorted(concepts)}."
        )
    new_ref = store.add(tm, parent=model_ref, operation="set_values")
    return summarize_model(tm, new_ref)


def export_model(store: ModelStore, model_ref: str, fmt: str = "template_model") -> dict:
    """Return a model's full JSON as a TemplateModel or ASKEM Petri net AMR."""
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"Unknown format {fmt!r}; choose one of {EXPORT_FORMATS}.")
    if fmt == "template_model":
        return store.get_json(model_ref)
    return template_model_to_petrinet_json(store.get(model_ref))
