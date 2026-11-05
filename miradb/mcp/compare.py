"""Model comparison and graph drawing, built on ``mira.metamodel.comparison``.

Concepts are compared "with context": two concepts match when their
groundings and context (e.g. age group) agree. With ``refinement="dkg"`` a
concept can also be a refinement of another, when its grounding is an
ontological child of the other's (see ``miradb.mcp.refinement``).
"""

import logging
import re
import uuid
from pathlib import Path

from mira.metamodel.comparison import (
    ModelComparisonGraphdata,
    TemplateModelComparison,
    TemplateModelDelta,
    get_concept_comparison_table,
)

from miradb.mcp.mira_tools import _concept_summary, describe_template
from miradb.mcp.model_store import ModelStore
from miradb.mcp.refinement import RefinementMode, get_refinement_func

logger = logging.getLogger(__name__)

IMAGE_DIR = Path(__file__).parent / "images"
IMAGE_FORMATS = ("png", "svg")
# Larger images are saved but not returned inline
MAX_INLINE_IMAGE_BYTES = 900_000
DELTA_LEGEND = {
    "blue": "only in model 1",
    "green": "only in model 2",
    "orange": "in both models",
    "red edges": "is_equal / refinement_of between the models",
}


def _with_warning(result: dict, warning: str | None) -> dict:
    if warning:
        result.setdefault("warnings", []).append(warning)
    return result


def compare_models(
    store: ModelStore,
    model_refs: list[str],
    refinement: RefinementMode = "dkg",
    include_graph: bool = False,
) -> dict:
    """Pairwise similarity of models (``ModelComparisonGraphdata``)."""
    if len(model_refs) < 2:
        raise ValueError("Provide at least two model_refs to compare.")
    refs = [store.normalize_ref(r) for r in model_refs]
    func, mode, warning = get_refinement_func(refinement)
    graph_data = ModelComparisonGraphdata.from_template_models(
        [store.get(r) for r in refs], refinement_func=func
    )
    scores = []
    for score in graph_data.get_similarity_scores():
        i, j = score["models"]
        scores.append({"models": [refs[i], refs[j]], "score": score["score"]})
    result = {"model_refs": refs, "refinement": mode, "similarity_scores": scores}
    if include_graph:
        # Model indices in the graph follow the order of model_refs
        result["graph"] = graph_data.to_json()
    return _with_warning(result, warning)


def _concept_label(concept, name_only: bool) -> str:
    """Label a concept the way ``get_concept_comparison_table`` does."""
    name = concept.display_name or concept.name or "N/A"
    if name_only:
        return name
    if concept.get_curie():
        name += f" ({':'.join(concept.get_curie())})"
    if concept.context:
        name += " [" + ", ".join(f"{k}: {v}" for k, v in concept.context.items()) + "]"
    return name


def compare_concepts(
    store: ModelStore,
    model_ref_1: str,
    model_ref_2: str,
    refinement: RefinementMode = "dkg",
    name_only: bool = False,
) -> dict:
    """Relate every concept of one model to every concept of another
    (``get_concept_comparison_table``)."""
    ref1, ref2 = store.normalize_ref(model_ref_1), store.normalize_ref(model_ref_2)
    tm1, tm2 = store.get(ref1), store.get(ref2)
    func, mode, warning = get_refinement_func(refinement)
    table = get_concept_comparison_table(
        tm1, tm2, refinement_func=func, name_only=name_only,
    )
    relation_names = {
        "=": "is_equal",
        ">": "refinement_of",  # model 1 concept refines model 2 concept
        "<": "refined_by",
    }
    # The table has model 1 concepts as columns and model 2 as rows
    relations = [
        {"model_1_concept": c1, "model_2_concept": c2,
         "relation": relation_names[symbol]}
        for c1 in table.columns
        for c2, symbol in table[c1].items()
        if symbol
    ]
    # The table only has rows/columns for concepts that matched something
    matched_1 = {r["model_1_concept"] for r in relations}
    matched_2 = {r["model_2_concept"] for r in relations}
    labels_1 = {_concept_label(c, name_only) for c in tm1.get_concepts_map().values()}
    labels_2 = {_concept_label(c, name_only) for c in tm2.get_concepts_map().values()}
    result = {
        "model_refs": [ref1, ref2],
        "refinement": mode,
        "relations": relations,
        "unmatched_in_model_1": sorted(labels_1 - matched_1),
        "unmatched_in_model_2": sorted(labels_2 - matched_2),
        "note": "refinement_of means the model 1 concept is a more specific "
                "version of the model 2 concept; refined_by the reverse.",
    }
    return _with_warning(result, warning)


def compare_context(store: ModelStore, model_refs: list[str]) -> dict:
    """Count the distinct values of each context key (e.g. age, location)
    per model (``TemplateModelComparison.compare_context``)."""
    if len(model_refs) < 2:
        raise ValueError("Provide at least two model_refs to compare.")
    refs = [store.normalize_ref(r) for r in model_refs]
    comparison = TemplateModelComparison(
        [store.get(r) for r in refs],
        refinement_func=lambda a, b: False,  # unused by compare_context
        tags=refs,
        run_on_init=False,
    )
    df = comparison.compare_context()
    return {
        "model_refs": refs,
        "context_value_counts": {
            key: {ref: (int(v) if v != "" else 0) for ref, v in row.items()}
            for key, row in df.iterrows()
        },
    }


def model_delta(
    store: ModelStore,
    model_ref_1: str,
    model_ref_2: str,
    refinement: RefinementMode = "dkg",
    concepts_only: bool = False,
) -> dict:
    """Describe what differs between two models, as text.

    Uses the same equality and refinement checks as ``TemplateModelDelta``,
    which ``draw_model_delta`` renders as a graph.
    """
    ref1, ref2 = store.normalize_ref(model_ref_1), store.normalize_ref(model_ref_2)
    tm1, tm2 = store.get(ref1), store.get(ref2)
    func, mode, warning = get_refinement_func(refinement)

    def relate(a, b):
        if a.is_equal_to(b, with_context=True):
            return "is_equal"
        if a.refinement_of(b, refinement_func=func, with_context=True):
            return "refinement_of"
        if b.refinement_of(a, refinement_func=func, with_context=True):
            return "refined_by"
        return None

    concepts1 = list(tm1.get_concepts_map().values())
    concepts2 = list(tm2.get_concepts_map().values())
    concept_matches, matched1, matched2 = [], set(), set()
    for i, c1 in enumerate(concepts1):
        for j, c2 in enumerate(concepts2):
            if relation := relate(c1, c2):
                concept_matches.append({"model_1": c1.name, "model_2": c2.name,
                                        "relation": relation})
                matched1.add(i)
                matched2.add(j)
    result = {
        "model_refs": [ref1, ref2],
        "refinement": mode,
        "concepts": {
            "matched": concept_matches,
            "only_in_model_1": [_concept_summary(c) for i, c in enumerate(concepts1)
                                if i not in matched1],
            "only_in_model_2": [_concept_summary(c) for j, c in enumerate(concepts2)
                                if j not in matched2],
        },
    }

    if not concepts_only:
        template_matches, matched1, matched2 = [], set(), set()
        for i, t1 in enumerate(tm1.templates):
            for j, t2 in enumerate(tm2.templates):
                if relation := relate(t1, t2):
                    template_matches.append({
                        "model_1": describe_template(t1, i),
                        "model_2": describe_template(t2, j),
                        "relation": relation,
                    })
                    matched1.add(i)
                    matched2.add(j)
        result["transitions"] = {
            "matched": template_matches,
            "only_in_model_1": [describe_template(t, i)
                                for i, t in enumerate(tm1.templates)
                                if i not in matched1],
            "only_in_model_2": [describe_template(t, j)
                                for j, t in enumerate(tm2.templates)
                                if j not in matched2],
            "note": "Transitions match on type and concepts; rate laws are "
                    "not compared.",
        }
    return _with_warning(result, warning)


def _image_path(stem: str, fmt: str) -> Path:
    if fmt not in IMAGE_FORMATS:
        raise ValueError(f"Unknown format {fmt!r}; choose from {IMAGE_FORMATS}.")
    IMAGE_DIR.mkdir(exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return IMAGE_DIR / f"{stem}_{uuid.uuid4().hex[:6]}.{fmt}"


def image_result(path: Path, info: dict) -> tuple[bytes | None, dict]:
    """Return the image bytes to inline (None if SVG or too large) and the
    info dict with the saved path."""
    info = {**info, "image_path": str(path)}
    if path.suffix != ".png":
        return None, info
    data = path.read_bytes()
    if len(data) > MAX_INLINE_IMAGE_BYTES:
        info["note"] = "Image too large to show inline; open image_path."
        return None, info
    return data, info


def draw_model(
    store: ModelStore,
    model_ref: str,
    use_display_name: bool = False,
    fmt: str = "png",
) -> tuple[bytes | None, dict]:
    """Render a model's graph (``TemplateModel.draw_graph``)."""
    ref = store.normalize_ref(model_ref)
    path = _image_path(ref, fmt)
    store.get(ref).draw_graph(str(path), use_display_name=use_display_name,
                              format=fmt, args="-Gdpi=96")
    return image_result(path, {"model_ref": ref})


def draw_model_delta(
    store: ModelStore,
    model_ref_1: str,
    model_ref_2: str,
    refinement: RefinementMode = "dkg",
    concepts_only: bool = False,
    fmt: str = "png",
) -> tuple[bytes | None, dict]:
    """Render the difference of two models (``TemplateModelDelta``)."""
    ref1, ref2 = store.normalize_ref(model_ref_1), store.normalize_ref(model_ref_2)
    func, mode, warning = get_refinement_func(refinement)
    delta = TemplateModelDelta(
        store.get(ref1), store.get(ref2), refinement_function=func,
        tag1=ref1, tag2=ref2, concepts_only=concepts_only,
    )
    path = _image_path(f"{ref1}_vs_{ref2}", fmt)
    delta.draw_graph(str(path), format=fmt, args="-Gdpi=96")
    info = {
        "model_refs": [ref1, ref2],
        "refinement": mode,
        "legend": {**DELTA_LEGEND,
                   "blue": f"only in {ref1}", "green": f"only in {ref2}"},
    }
    return image_result(path, _with_warning(info, warning))


def model_graph_json(store: ModelStore, model_ref: str) -> dict:
    """Node-link JSON of a model's graph (``TemplateModel.graph_as_json``)."""
    return store.get(model_ref).graph_as_json()
