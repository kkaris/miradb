"""Editing TemplateModels: components, metadata and structure.

These are the TemplateModel counterparts of ``mira.modeling.amr.ops``
(which edits AMR JSON) plus metadata edits from
``mira.metamodel.template_model``. Every function loads a fresh copy of the
input model, edits it, stores the result as a new workspace model and
returns a short report of what changed, never the whole model.

Renames propagate: renaming a state, parameter or the time variable also
renames its symbol in rate laws, observables and initial expressions.
"""

import copy
import datetime
import logging

import sympy

from mira.metamodel import TemplateModel
from mira.metamodel.ops import aggregate_parameters, counts_to_dimensionless
from mira.metamodel.template_model import (
    Annotations,
    Author,
    Distribution,
    Initial,
    Observable,
    Parameter,
    Time,
)
from mira.metamodel.templates import (
    Concept,
    ControlledConversion,
    ControlledDegradation,
    ControlledProduction,
    GroupedControlledConversion,
    GroupedControlledDegradation,
    GroupedControlledProduction,
    NaturalConversion,
    NaturalDegradation,
    NaturalProduction,
    StaticConcept,
)
from mira.metamodel.units import Unit

from miradb.mcp.mira_tools import (
    _template_concepts,
    describe_template,
    find_transition,
    parse_expression,
    transition_id,
    undeclared_rate_symbols,
)
from miradb.mcp.model_store import ModelStore

logger = logging.getLogger(__name__)

ANNOTATION_LIST_FIELDS = (
    "references", "locations", "pathogens", "diseases", "hosts", "model_types",
)
ANNOTATION_TEXT_FIELDS = ("name", "description", "license", "time_scale")

# (has subject, has outcome, number of controllers) -> template class
_TEMPLATE_CLASSES = {
    (True, True, 0): NaturalConversion,
    (True, True, 1): ControlledConversion,
    (True, True, 2): GroupedControlledConversion,
    (False, True, 0): NaturalProduction,
    (False, True, 1): ControlledProduction,
    (False, True, 2): GroupedControlledProduction,
    (True, False, 0): NaturalDegradation,
    (True, False, 1): ControlledDegradation,
    (True, False, 2): GroupedControlledDegradation,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _model_warnings(tm: TemplateModel) -> list[str]:
    warnings = []
    undeclared = undeclared_rate_symbols(tm)
    if undeclared:
        warnings.append(
            f"Rate laws use undeclared symbols {sorted(undeclared)}; add them "
            "with mira_add_parameter."
        )
    known = (set(tm.get_concepts_name_map()) | set(tm.parameters)
             | ({tm.time.name} if tm.time else set()))
    for name, obs in tm.observables.items():
        unknown = {str(s) for s in obs.expression.free_symbols} - known
        if unknown:
            warnings.append(f"Observable {name!r} uses unknown symbols "
                            f"{sorted(unknown)}.")
    return warnings


def _commit(
    store: ModelStore,
    tm: TemplateModel,
    parent: str,
    operation: str,
    changes: dict | None = None,
) -> dict:
    """Store an edited model and return the report tools send back."""
    new_ref = store.add(tm, parent=parent, operation=operation)
    result = {
        "model_ref": new_ref,
        "parent": store.normalize_ref(parent),
        "operation": operation,
    }
    if changes:
        result["changes"] = changes
    warnings = _model_warnings(tm)
    if warnings:
        result["warnings"] = warnings
    return result


def _state_names(tm: TemplateModel) -> set[str]:
    return set(tm.get_concepts_name_map())


def _require_state(tm: TemplateModel, name: str):
    if name not in _state_names(tm):
        raise ValueError(
            f"No state {name!r}. States: {sorted(_state_names(tm))}."
        )


def _require_parameter(tm: TemplateModel, name: str):
    if name not in tm.parameters:
        raise ValueError(
            f"No parameter {name!r}. Parameters: {sorted(tm.parameters)}."
        )


def _require_observable(tm: TemplateModel, name: str):
    if name not in tm.observables:
        raise ValueError(
            f"No observable {name!r}. Observables: {sorted(tm.observables)}."
        )


def _require_new_symbol(tm: TemplateModel, name: str):
    """Refuse a name that would collide with an existing symbol."""
    if name in _state_names(tm):
        raise ValueError(f"{name!r} is already a state.")
    if name in tm.parameters:
        raise ValueError(f"{name!r} is already a parameter.")
    if name in tm.observables:
        raise ValueError(f"{name!r} is already an observable.")
    if tm.time and name == tm.time.name:
        raise ValueError(f"{name!r} is the time variable.")


def _rename_symbol(tm: TemplateModel, old: str, new: str):
    """Rename a symbol in every expression of the model."""
    subs = {sympy.Symbol(old): sympy.Symbol(new)}
    for template in tm.templates:
        if template.rate_law is not None:
            template.rate_law = template.rate_law.xreplace(subs)
    for obs in tm.observables.values():
        obs.expression = sympy.sympify(obs.expression).xreplace(subs)
    for initial in tm.initials.values():
        initial.expression = sympy.sympify(initial.expression).xreplace(subs)


def _parse_units(units: str | None) -> Unit | None:
    return Unit(parse_expression(units)) if units else None


def _parse_distribution(distribution: dict | None) -> Distribution | None:
    if not distribution:
        return None
    if "type" not in distribution:
        raise ValueError("A distribution needs a 'type', e.g. "
                         "{'type': 'Uniform1', 'parameters': {'minimum': 0, "
                         "'maximum': 1}}.")
    return Distribution(type=distribution["type"],
                        parameters=distribution.get("parameters") or {})


def _state_concepts(tm: TemplateModel, name: str):
    """Yield every Concept object of the model that represents a state.

    MIRA stores a separate copy of a state's Concept in each transition
    (and in its initial), so edits must reach all of them.
    """
    for template in tm.templates:
        for concept in _template_concepts(template):
            if concept.name == name:
                yield concept
    for initial in tm.initials.values():
        if initial.concept.name == name:
            yield initial.concept


def _annotations(tm: TemplateModel) -> Annotations:
    if tm.annotations is None:
        tm.annotations = Annotations()
    return tm.annotations


def _unused_parameters(tm: TemplateModel) -> list[str]:
    return sorted(set(tm.parameters) - tm.get_all_used_parameters())


def _define_rate_law_symbols(
    tm: TemplateModel, rate_law: sympy.Expr, parameters: dict | None
) -> tuple[list[str], list[str]]:
    """Add parameters for a rate law's free symbols that are not yet
    declared, using the attributes in ``parameters`` where given.

    Returns the names added with and without a value.
    """
    known = _state_names(tm) | set(tm.parameters) | {"t"}
    if tm.time:
        known.add(tm.time.name)
    with_value, without_value = [], []
    for symbol in sorted(str(s) for s in rate_law.free_symbols):
        if symbol in known:
            continue
        attrs = (parameters or {}).get(symbol, {})
        tm.parameters[symbol] = _new_parameter(symbol, **attrs)
        (with_value if attrs.get("value") is not None else without_value).append(symbol)
    return with_value, without_value


def _new_parameter(
    name: str,
    value: float | None = None,
    distribution: dict | None = None,
    units: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> Parameter:
    return Parameter(
        name=name,
        value=float(value) if value is not None else None,
        distribution=_parse_distribution(distribution),
        units=_parse_units(units),
        display_name=display_name,
        description=description,
    )


# ---------------------------------------------------------------------------
# States (concepts)
# ---------------------------------------------------------------------------

def add_state(
    store: ModelStore,
    model_ref: str,
    name: str,
    display_name: str | None = None,
    description: str | None = None,
    identifiers: dict[str, str] | None = None,
    context: dict[str, str] | None = None,
    units: str | None = None,
    initial: float | str | None = None,
) -> dict:
    """Add a state that no transition uses yet (a StaticConcept)."""
    tm = store.get(model_ref)
    _require_new_symbol(tm, name)
    concept = Concept(
        name=name, display_name=display_name, description=description,
        identifiers=identifiers, context=context, units=_parse_units(units),
    )
    tm.templates.append(StaticConcept(subject=concept))
    if initial is not None:
        tm.initials[name] = Initial(concept=copy.deepcopy(concept),
                                    expression=parse_expression(initial))
    return _commit(store, tm, model_ref, f"add_state({name!r})", {
        "added_state": name,
        "initial": str(initial) if initial is not None else None,
        "hint": "Connect the state with mira_add_transition.",
    })


def remove_state(store: ModelStore, model_ref: str, name: str) -> dict:
    """Remove a state, every transition involving it and its initial; its
    symbol is replaced by 0 in observables."""
    tm = store.get(model_ref)
    _require_state(tm, name)
    removed, kept = [], []
    for index, template in enumerate(tm.templates):
        if any(c.name == name for c in _template_concepts(template)):
            removed.append(describe_template(template, index))
        else:
            kept.append(template)
    tm.templates = kept
    tm.initials.pop(name, None)
    zero = {sympy.Symbol(name): sympy.Integer(0)}
    changed_observables = []
    for obs_name, obs in tm.observables.items():
        expr = sympy.sympify(obs.expression)
        if sympy.Symbol(name) in expr.free_symbols:
            obs.expression = expr.xreplace(zero)
            changed_observables.append(obs_name)
    return _commit(store, tm, model_ref, f"remove_state({name!r})", {
        "removed_state": name,
        "removed_transitions": removed,
        "observables_with_state_set_to_0": changed_observables,
        "parameters_now_unused": _unused_parameters(tm),
        "note": "Transition IDs of unnamed transitions (t<n>) shift after "
                "removals; use mira_get_model for the current IDs.",
    })


def update_state(
    store: ModelStore,
    model_ref: str,
    name: str,
    new_name: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
    identifiers: dict[str, str] | None = None,
    context: dict[str, str] | None = None,
    units: str | None = None,
) -> dict:
    """Edit a state's attributes and optionally rename it everywhere."""
    tm = store.get(model_ref)
    _require_state(tm, name)
    if new_name and new_name != name:
        _require_new_symbol(tm, new_name)
    new_units = _parse_units(units)
    changed = {k: v for k, v in {
        "display_name": display_name, "description": description,
        "identifiers": identifiers, "context": context, "units": units,
    }.items() if v is not None}
    for concept in _state_concepts(tm, name):
        if display_name is not None:
            concept.display_name = display_name
        if description is not None:
            concept.description = description
        if identifiers is not None:
            concept.identifiers = dict(identifiers)
        if context is not None:
            concept.context = dict(context)
        if new_units is not None:
            concept.units = new_units
        if new_name and new_name != name:
            concept.name = new_name
    if new_name and new_name != name:
        _rename_symbol(tm, name, new_name)
        if name in tm.initials:
            tm.initials[new_name] = tm.initials.pop(name)
        changed["name"] = new_name
    if not changed:
        raise ValueError("Nothing to change: pass at least one attribute.")
    return _commit(store, tm, model_ref, f"update_state({name!r})",
                   {"state": name, "set": changed})


# ---------------------------------------------------------------------------
# Initials
# ---------------------------------------------------------------------------

def set_initial(
    store: ModelStore, model_ref: str, state: str, expression: float | str
) -> dict:
    """Add or replace the initial condition of a state."""
    tm = store.get(model_ref)
    _require_state(tm, state)
    concept = copy.deepcopy(tm.get_concepts_name_map()[state])
    previous = tm.initials.get(state)
    tm.initials[state] = Initial(concept=concept,
                                 expression=parse_expression(expression))
    return _commit(store, tm, model_ref, f"set_initial({state!r})", {
        "state": state,
        "initial": str(tm.initials[state].expression),
        "previous": str(previous.expression) if previous else None,
    })


def remove_initial(store: ModelStore, model_ref: str, state: str) -> dict:
    tm = store.get(model_ref)
    if state not in tm.initials:
        raise ValueError(f"State {state!r} has no initial. Initials: "
                         f"{sorted(tm.initials)}.")
    removed = tm.initials.pop(state)
    return _commit(store, tm, model_ref, f"remove_initial({state!r})",
                   {"state": state, "removed": str(removed.expression)})


# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

def add_parameter(
    store: ModelStore,
    model_ref: str,
    name: str,
    value: float | None = None,
    distribution: dict | None = None,
    units: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    tm = store.get(model_ref)
    _require_new_symbol(tm, name)
    tm.parameters[name] = _new_parameter(
        name, value, distribution, units, display_name, description
    )
    return _commit(store, tm, model_ref, f"add_parameter({name!r})",
                   {"added_parameter": name, "value": value})


def remove_parameter(
    store: ModelStore,
    model_ref: str,
    name: str,
    replacement_value: float | None = None,
) -> dict:
    """Remove a parameter, substituting ``replacement_value`` (default 0)
    wherever it is used."""
    tm = store.get(model_ref)
    _require_parameter(tm, name)
    value = replacement_value if replacement_value is not None else 0
    tm.substitute_parameter(name, value=value)
    zeroed = [transition_id(t, i) for i, t in enumerate(tm.templates)
              if t.rate_law is not None and t.rate_law == 0]
    changes = {"removed_parameter": name, "substituted_value": value}
    if zeroed:
        changes["transitions_with_zero_rate"] = zeroed
    return _commit(store, tm, model_ref, f"remove_parameter({name!r})", changes)


def update_parameter(
    store: ModelStore,
    model_ref: str,
    name: str,
    new_name: str | None = None,
    value: float | None = None,
    distribution: dict | None = None,
    clear_distribution: bool = False,
    units: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    """Edit a parameter's attributes and optionally rename it everywhere."""
    tm = store.get(model_ref)
    _require_parameter(tm, name)
    param = tm.parameters[name]
    changed = {}
    if value is not None:
        param.value = float(value)
        changed["value"] = value
    if distribution is not None:
        param.distribution = _parse_distribution(distribution)
        changed["distribution"] = distribution
    elif clear_distribution:
        param.distribution = None
        changed["distribution"] = None
    if units is not None:
        param.units = _parse_units(units)
        changed["units"] = units
    if display_name is not None:
        param.display_name = display_name
        changed["display_name"] = display_name
    if description is not None:
        param.description = description
        changed["description"] = description
    if new_name and new_name != name:
        _require_new_symbol(tm, new_name)
        _rename_symbol(tm, name, new_name)
        param.name = new_name
        tm.parameters = {(new_name if k == name else k): v
                         for k, v in tm.parameters.items()}
        changed["name"] = new_name
    if not changed:
        raise ValueError("Nothing to change: pass at least one attribute.")
    return _commit(store, tm, model_ref, f"update_parameter({name!r})",
                   {"parameter": name, "set": changed})


def eliminate_unused_parameters(store: ModelStore, model_ref: str) -> dict:
    tm = store.get(model_ref)
    unused = _unused_parameters(tm)
    tm.eliminate_unused_parameters()
    # MIRA only checks rate laws here; keep anything initials/observables use
    return _commit(store, tm, model_ref, "eliminate_unused_parameters",
                   {"removed_parameters": sorted(set(unused) - set(tm.parameters))})


def merge_duplicate_parameters(
    store: ModelStore, model_ref: str, redundant: str, keep: str
) -> dict:
    """Replace parameter ``redundant`` by ``keep`` everywhere and drop it."""
    tm = store.get(model_ref)
    _require_parameter(tm, redundant)
    _require_parameter(tm, keep)
    if redundant == keep:
        raise ValueError("redundant and keep must differ.")
    _rename_symbol(tm, redundant, keep)
    tm.parameters.pop(redundant)
    return _commit(store, tm, model_ref,
                   f"merge_duplicate_parameters({redundant!r} -> {keep!r})",
                   {"removed_parameter": redundant, "replaced_by": keep})


def aggregate_model_parameters(store: ModelStore, model_ref: str) -> dict:
    """Combine parameter products in mass-action rate laws into single
    parameters (``mira.metamodel.ops.aggregate_parameters``)."""
    tm = store.get(model_ref)
    # MIRA folds parameter values into numbers and fails on missing ones
    missing = sorted(n for n, p in tm.parameters.items() if p.value is None)
    if missing:
        raise ValueError(
            f"Parameters {missing} have no value; aggregation multiplies "
            "parameter values together. Set them first (mira_set_values or "
            "mira_apply_scenario)."
        )
    before = set(tm.parameters)
    new_tm = aggregate_parameters(tm)
    after = set(new_tm.parameters)
    return _commit(store, new_tm, model_ref, "aggregate_parameters", {
        "added_parameters": sorted(after - before),
        "removed_parameters": sorted(before - after),
    })


def to_dimensionless(
    store: ModelStore, model_ref: str, counts_unit: str, norm_factor: float
) -> dict:
    """Convert quantities in ``counts_unit`` (e.g. 'person') to fractions
    (``mira.metamodel.ops.counts_to_dimensionless``)."""
    new_tm = counts_to_dimensionless(store.get(model_ref), counts_unit,
                                     norm_factor)
    return _commit(store, new_tm, model_ref,
                   f"counts_to_dimensionless({counts_unit!r}, {norm_factor})",
                   {"counts_unit": counts_unit, "norm_factor": norm_factor})


# ---------------------------------------------------------------------------
# Transitions
# ---------------------------------------------------------------------------

def add_transition(
    store: ModelStore,
    model_ref: str,
    name: str | None = None,
    subject: str | None = None,
    outcome: str | None = None,
    controllers: list[str] | None = None,
    rate_law: str | None = None,
    mass_action_parameter: str | None = None,
    parameters: dict[str, dict] | None = None,
) -> dict:
    """Add a transition; its type follows from which states are given.

    subject + outcome: conversion; outcome only: production; subject only:
    degradation; with controllers, the controlled (one controller) or
    grouped-controlled (several) variant. Without a rate law, a mass-action
    rate law with ``mass_action_parameter`` is used.
    """
    tm = store.get(model_ref)
    controllers = controllers or []
    if not subject and not outcome:
        raise ValueError("Give a subject, an outcome, or both.")
    for state in [s for s in (subject, outcome) if s] + controllers:
        _require_state(tm, state)
    if name and any(t.name == name for t in tm.templates):
        raise ValueError(f"A transition named {name!r} already exists.")

    concepts = tm.get_concepts_name_map()
    key = (bool(subject), bool(outcome), min(len(controllers), 2))
    cls = _TEMPLATE_CLASSES[key]
    kwargs = {"name": name}
    if subject:
        kwargs["subject"] = copy.deepcopy(concepts[subject])
    if outcome:
        kwargs["outcome"] = copy.deepcopy(concepts[outcome])
    if len(controllers) == 1:
        kwargs["controller"] = copy.deepcopy(concepts[controllers[0]])
    elif controllers:
        kwargs["controllers"] = [copy.deepcopy(concepts[c]) for c in controllers]
    template = cls(**kwargs)

    if rate_law:
        template.rate_law = parse_expression(rate_law)
    else:
        param = mass_action_parameter or f"k_{name or f't{len(tm.templates) + 1}'}"
        template.set_mass_action_rate_law(param)
    with_value, without_value = _define_rate_law_symbols(
        tm, template.rate_law, parameters
    )
    tm.templates.append(template)
    changes = {"added_transition": describe_template(template,
                                                     len(tm.templates) - 1)}
    if with_value:
        changes["added_parameters"] = with_value
    if without_value:
        changes["added_parameters_without_value"] = without_value
    return _commit(store, tm, model_ref, f"add_transition({cls.__name__})",
                   changes)


def remove_transition(store: ModelStore, model_ref: str, transition: str) -> dict:
    tm = store.get(model_ref)
    index = find_transition(tm, transition)
    removed = describe_template(tm.templates[index], index)
    tm.templates.pop(index)
    return _commit(store, tm, model_ref, f"remove_transition({removed['id']!r})", {
        "removed_transition": removed,
        "parameters_now_unused": _unused_parameters(tm),
        "note": "Transition IDs of unnamed transitions (t<n>) shift after "
                "removals; use mira_get_model for the current IDs.",
    })


def update_transition(
    store: ModelStore,
    model_ref: str,
    transition: str,
    new_name: str | None = None,
    rate_law: str | None = None,
    display_name: str | None = None,
    parameters: dict[str, dict] | None = None,
) -> dict:
    """Rename a transition and/or replace its rate law."""
    tm = store.get(model_ref)
    index = find_transition(tm, transition)
    template = tm.templates[index]
    changed = {}
    if new_name:
        if any(t.name == new_name for i, t in enumerate(tm.templates)
               if i != index):
            raise ValueError(f"A transition named {new_name!r} already exists.")
        template.name = new_name
        changed["name"] = new_name
    if display_name is not None:
        template.display_name = display_name
        changed["display_name"] = display_name
    if rate_law:
        changed["previous_rate_law"] = str(template.rate_law)
        template.rate_law = parse_expression(rate_law)
        changed["rate_law"] = str(template.rate_law)
        with_value, without_value = _define_rate_law_symbols(
            tm, template.rate_law, parameters
        )
        if with_value:
            changed["added_parameters"] = with_value
        if without_value:
            changed["added_parameters_without_value"] = without_value
        changed["parameters_now_unused"] = _unused_parameters(tm)
    if not changed:
        raise ValueError("Nothing to change: pass new_name, rate_law or "
                         "display_name.")
    return _commit(store, tm, model_ref, f"update_transition({transition!r})",
                   {"transition": describe_template(template, index), "set": changed})


# ---------------------------------------------------------------------------
# Observables
# ---------------------------------------------------------------------------

def add_observable(
    store: ModelStore,
    model_ref: str,
    name: str,
    expression: str,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    tm = store.get(model_ref)
    _require_new_symbol(tm, name)
    tm.observables[name] = Observable(
        name=name, expression=parse_expression(expression),
        display_name=display_name, description=description,
    )
    return _commit(store, tm, model_ref, f"add_observable({name!r})", {
        "added_observable": name,
        "expression": str(tm.observables[name].expression),
    })


def remove_observable(store: ModelStore, model_ref: str, name: str) -> dict:
    tm = store.get(model_ref)
    _require_observable(tm, name)
    removed = tm.observables.pop(name)
    return _commit(store, tm, model_ref, f"remove_observable({name!r})", {
        "removed_observable": name, "expression": str(removed.expression),
    })


def update_observable(
    store: ModelStore,
    model_ref: str,
    name: str,
    new_name: str | None = None,
    expression: str | None = None,
    display_name: str | None = None,
    description: str | None = None,
) -> dict:
    tm = store.get(model_ref)
    _require_observable(tm, name)
    obs = tm.observables[name]
    changed = {}
    if expression is not None:
        changed["previous_expression"] = str(obs.expression)
        obs.expression = parse_expression(expression)
        changed["expression"] = str(obs.expression)
    if display_name is not None:
        obs.display_name = display_name
        changed["display_name"] = display_name
    if description is not None:
        obs.description = description
        changed["description"] = description
    if new_name and new_name != name:
        _require_new_symbol(tm, new_name)
        obs.name = new_name
        tm.observables = {(new_name if k == name else k): v
                          for k, v in tm.observables.items()}
        changed["name"] = new_name
    if not changed:
        raise ValueError("Nothing to change: pass at least one attribute.")
    return _commit(store, tm, model_ref, f"update_observable({name!r})",
                   {"observable": name, "set": changed})


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

def set_annotations(store: ModelStore, model_ref: str, **fields) -> dict:
    """Set model-level annotations. Fields given replace the old value
    (lists included); fields omitted are kept. Authors have their own
    tools."""
    tm = store.get(model_ref)
    ann = _annotations(tm)
    changed = {}
    for key, value in fields.items():
        if value is None:
            continue
        if key in ("time_start", "time_end"):
            value = datetime.datetime.fromisoformat(value)
        elif key in ANNOTATION_LIST_FIELDS:
            value = list(value)
        elif key not in ANNOTATION_TEXT_FIELDS:
            raise ValueError(f"Unknown annotation field {key!r}.")
        setattr(ann, key, value)
        changed[key] = fields[key]
    if not changed:
        raise ValueError("Nothing to change: pass at least one field.")
    return _commit(store, tm, model_ref, "set_annotations",
                   {"set": changed, "annotations": ann.to_json()})


def add_author(store: ModelStore, model_ref: str, name: str) -> dict:
    tm = store.get(model_ref)
    ann = _annotations(tm)
    if any(a.name == name for a in ann.authors):
        raise ValueError(f"{name!r} is already an author.")
    ann.authors.append(Author(name=name))
    return _commit(store, tm, model_ref, f"add_author({name!r})",
                   {"authors": [a.name for a in ann.authors]})


def remove_author(store: ModelStore, model_ref: str, name: str) -> dict:
    tm = store.get(model_ref)
    ann = _annotations(tm)
    if not any(a.name == name for a in ann.authors):
        raise ValueError(f"{name!r} is not an author. Authors: "
                         f"{[a.name for a in ann.authors]}.")
    ann.authors = [a for a in ann.authors if a.name != name]
    return _commit(store, tm, model_ref, f"remove_author({name!r})",
                   {"authors": [a.name for a in ann.authors]})


def set_time(
    store: ModelStore,
    model_ref: str,
    name: str | None = None,
    units: str | None = None,
) -> dict:
    """Set the time variable's name (renaming it in rate laws) and units."""
    tm = store.get(model_ref)
    old = tm.time or Time(name="t", units=None)
    new_name = name or old.name
    if new_name != old.name:
        _require_new_symbol(tm, new_name)
        _rename_symbol(tm, old.name, new_name)
    tm.time = Time(name=new_name,
                   units=_parse_units(units) if units else old.units)
    return _commit(store, tm, model_ref, "set_time",
                   {"time": tm.time.to_json()})


def merge_models(store: ModelStore, model_ref: str, other_model_ref: str) -> dict:
    """Add another model's transitions, parameters, initials and observables
    (``TemplateModel.extend``). On name clashes the first model's
    definitions are kept."""
    tm = store.get(model_ref)
    other = store.get(other_model_ref)
    clashes = {
        "parameters": sorted(set(tm.parameters) & set(other.parameters)),
        "initials": sorted(set(tm.initials) & set(other.initials)),
        "observables": sorted(set(tm.observables) & set(other.observables)),
    }
    new_params = {k: v for k, v in other.parameters.items()
                  if k not in tm.parameters}
    new_initials = {k: v for k, v in other.initials.items()
                    if k not in tm.initials}
    merged = tm.extend(other, parameter_mapping=new_params,
                       initial_mapping=new_initials)
    for k, v in other.observables.items():
        merged.observables.setdefault(k, v)
    changes = {
        "added_transitions": len(other.templates),
        "added_parameters": sorted(new_params),
        "added_initials": sorted(new_initials),
        "kept_from_first_model": {k: v for k, v in clashes.items() if v},
        "note": "States with the same name are treated as the same state.",
    }
    return _commit(store, merged, model_ref,
                   f"merge_models(+{store.normalize_ref(other_model_ref)})",
                   changes)


def apply_db_scenario(store: ModelStore, model_ref: str, scenario: dict) -> dict:
    """Bake a DB scenario (from ``db_tools.get_scenario``) into a model:
    parameter values, plus initials from parameters like ``I_0``."""
    from miradb.mcp.simulate import apply_scenario

    tm = store.get(model_ref)
    report, _ = apply_scenario(tm, scenario)
    return _commit(store, tm, model_ref,
                   f"apply_scenario({scenario['scenario_id']})", report)
