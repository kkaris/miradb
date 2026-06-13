__all__ = [
    "compare_models",
    "term_set_jaccard",
    "compartment_jaccard",
    "tree_edit_distance",
]

import logging

import sympy
from sympy import simplify, fraction, expand
from sympy.core.function import AppliedUndef
from rapidfuzz import fuzz
import zss


logger = logging.getLogger(__name__)

#  ----------------------------------
# LAYER 1: COMPARTMENT JACCARD
#  ----------------------------------


def parse_ode_string(ode_str: str) -> dict:
    """Parse an ODE string to a mapping of variable names to RHS expressions

    Parameters
    ----------
    ode_str :
        The ODE string to execute and parse.

    Returns
    -------
    :
        Dictionary mapping variable names to their right-hand side sympy
        expressions.
    """
    ns = {"sympy": sympy}
    exec(ode_str, ns)

    t_canonical = sympy.Symbol("t")
    t_actual = ns.get("t") or ns.get("T") or t_canonical

    rhs_map = {}
    for eq in ns["odes"]:
        func = eq.lhs.args[0]
        var_name = func.func.__name__
        rhs = eq.rhs
        if t_actual != t_canonical:
            rhs = rhs.replace(
                lambda expr: isinstance(expr, AppliedUndef)
                and expr.args == (t_actual,),
                lambda expr: expr.func(t_canonical),
            )
        rhs_map[var_name] = rhs
    return rhs_map


def compartment_jaccard(
    rhs1: dict, rhs2: dict, mismatch_threshold: float = 0.5
) -> dict:
    """Compute the Jaccard index for compartments between two ODE models

    Parameters
    ----------
    rhs1 :
        Dictionary of variable names to sympy expressions for model 1.
    rhs2 :
        Dictionary of variable names to sympy expressions for model 2.
    mismatch_threshold :
        Threshold below which a mismatch is flagged (default is 0.5).

    Returns
    -------
    :
        Dictionary with Jaccard score, mismatch flag, shared/unique
        compartments, and aligned rhs2.
    """
    roles1, roles2 = set(rhs1), set(rhs2)
    intersection = roles1 & roles2
    union = roles1 | roles2
    score = len(intersection) / len(union) if union else 1.0
    r1_only = sorted(roles1 - roles2)
    r2_only = sorted(roles2 - roles1)

    subs = {}
    for comp1 in r1_only:
        for comp2 in r2_only:
            ratio = fuzz.ratio(comp1, comp2)
            if ratio > 80 and comp1 not in subs.values():
                subs[comp2] = comp1
                break

    rhs2_aligned = rhs2
    if subs:
        rhs2_aligned = {}
        t = sympy.Symbol("t")
        for comp, expr in rhs2.items():
            new_comp = subs.get(comp, comp)
            if comp in subs:
                old_sym = sympy.Function(comp)
                new_sym = sympy.Function(subs[comp])
                rhs2_aligned[new_comp] = expr.subs(old_sym(t), new_sym(t))
            else:
                rhs2_aligned[new_comp] = expr

    return {
        "jaccard": score,
        "compartment_mismatch": score < mismatch_threshold,
        "shared": sorted(intersection),
        "only_in_1": r1_only,
        "only_in_2": r2_only,
        "rhs2_aligned": rhs2_aligned,
    }


#  ----------------------------------
# LAYER 2: TERM-SET JACCARD (PER-COMPARTMENT)
#  ----------------------------------

def get_terms(canon_expr) -> list:
    """Split expression into terms, stripping scalar magnitude, preserving sign

    Parameters
    ----------
    canon_expr : sympy.Expr
        Canonicalized sympy expression to split into terms.

    Returns
    -------
    :
        List of sympy expressions representing the terms.
    """
    result = []
    numer, denom = fraction(canon_expr)
    terms = [t / denom for t in expand(numer).as_ordered_terms()]
    for term in terms:
        coeff, structural = term.as_coeff_Mul()
        sign = sympy.Integer(-1) if coeff < 0 else sympy.Integer(1)
        result.append(sign * structural)
    return result


def terms_match(t1, t2) -> bool:
    """Check if two sympy terms are mathematically equivalent.

    Parameters
    ----------
    t1 : sympy.Expr
        First term.
    t2 : sympy.Expr
        Second term.

    Returns
    -------
    :
        True if terms are equivalent, False otherwise.
    """
    return simplify(t1 - t2) == 0


def term_jaccard_per_compartment(terms1: list, terms2: list) -> dict:
    """Compute the Jaccard index for sets of terms in a compartment.

    Parameters
    ----------
    terms1 :
        List of sympy terms for compartment 1.
    terms2 :
        List of sympy terms for compartment 2.

    Returns
    -------
    :
        Jaccard score, shared/unique terms, and no_shared_terms flag.
    """
    matched1, matched2 = set(), set()
    for i, t1 in enumerate(terms1):
        for j, t2 in enumerate(terms2):
            if j not in matched2 and terms_match(t1, t2):
                matched1.add(i)
                matched2.add(j)
                break

    n_shared = len(matched1)
    n_union = len(terms1) + len(terms2) - n_shared
    jaccard = n_shared / n_union if n_union else 1.0
    return {
        "jaccard": jaccard,
        "no_shared_terms": n_shared == 0,
        "only_in_1": [
            terms1[i] for i in range(len(terms1)) if i not in matched1
        ],
        "only_in_2": [
            terms2[j] for j in range(len(terms2)) if j not in matched2
        ],
    }


def term_set_jaccard(canon1: dict, canon2: dict) -> dict:
    """
    Compute the Jaccard index for term sets across all shared compartments.

    Parameters
    ----------
    canon1 :
        Canonicalized compartment-to-expression mapping for model 1.
    canon2 :
        Canonicalized compartment-to-expression mapping for model 2.

    Returns
    -------
    :
        Per-compartment and aggregate Jaccard scores, and unique compartments.
    """
    shared = sorted(set(canon1) & set(canon2))
    only_in1 = sorted(set(canon1) - set(canon2))
    only_in2 = sorted(set(canon2) - set(canon1))

    per_compartment, scores = {}, []
    for role in shared:
        terms1 = get_terms(canon1[role])
        terms2 = get_terms(canon2[role])
        result = term_jaccard_per_compartment(terms1, terms2)
        per_compartment[role] = result
        scores.append(result["jaccard"])

    return {
        "per_compartment": per_compartment,
        "aggregate": sum(scores) / len(scores) if scores else 0.0,
        "only_in_1": only_in1,
        "only_in_2": only_in2,
    }


#  ----------------------------------
# LAYER 3: TREE EDIT DISTANCE
#  ----------------------------------

def expr_to_tree(expr) -> zss.Node:
    """Convert a sympy expression to a zss.Node for tree edit distance.

    Parameters
    ----------
    expr : sympy.Expr
        Sympy expression to convert.

    Returns
    -------
    :
        Root node of the tree representation.
    """
    if isinstance(expr, AppliedUndef):
        var_name = expr.func.__name__
        return zss.Node(var_name)

    if not expr.args:
        return zss.Node(str(expr))

    node = zss.Node(type(expr).__name__)
    for arg in expr.args:
        node.addkid(expr_to_tree(arg))
    return node


def tree_size(node: zss.Node) -> int:
    """Recursively count the number of nodes in a zss tree.

    Parameters
    ----------
    node :
        Root node of the tree.

    Returns
    -------
    :
        Number of nodes in the tree.
    """
    return 1 + sum(tree_size(c) for c in node.children)


def ted_for_pair(e1, e2) -> dict:
    """Compute the tree edit distance (TED) between two sympy expressions.

    Parameters
    ----------
    e1 : sympy.Expr
        First expression.
    e2 : sympy.Expr
        Second expression.

    Returns
    -------
    :
        Raw and normalized TED scores.
    """
    t1 = expr_to_tree(e1)
    t2 = expr_to_tree(e2)
    raw = zss.simple_distance(t1, t2)
    norm = raw / max(tree_size(t1), tree_size(t2))
    return {"raw": raw, "normalized": round(norm, 4)}


def tree_edit_distance(canon1: dict, canon2: dict) -> dict:
    """Compute tree edit distance metrics for shared compartments

    Parameters
    ----------
    canon1 :
        Canonicalized compartment-to-expression mapping for model 1.
    canon2 :
        Canonicalized compartment-to-expression mapping for model 2.

    Returns
    -------
    :
        Per-compartment, aggregate, and whole-model TED scores.
    """
    shared  = sorted(set(canon1) & set(canon2))

    per_compartment = {}
    raw_scores, norm_scores = [], []

    for role in shared:
        scores = ted_for_pair(canon1[role], canon2[role])
        per_compartment[role] = scores
        raw_scores.append(scores["raw"])
        norm_scores.append(scores["normalized"])

    # Whole-model TED: combine all compartment RHS into one expression
    whole1 = sympy.Add(*[canon1[r] for r in shared])
    whole2 = sympy.Add(*[canon2[r] for r in shared])
    wt1 = expr_to_tree(whole1)
    wt2 = expr_to_tree(whole2)
    w_raw = zss.simple_distance(wt1, wt2)
    w_norm = w_raw / max(tree_size(wt1), tree_size(wt2))

    results = {
        "per_compartment": per_compartment,
        "aggregate_per_compartment": {
            "raw": round(sum(raw_scores) / len(raw_scores), 2) if raw_scores else 0,
            "normalized": (
                round(sum(norm_scores) / len(norm_scores), 4) if norm_scores else 0
            ),
        },
        "whole_model": {
            "raw": w_raw,
            "normalized": round(w_norm, 4)
        },
    }

    return results


def get_params(
    rhs_dict: dict,
    state_var_names: set,
    time_symbol=sympy.Symbol("t")
) -> set:
    """Extract parameter symbols from RHS expressions, ignoring states and time

    Parameters
    ----------
    rhs_dict :
        Mapping of variable names to sympy expressions.
    state_var_names :
        Set of state variable names.
    time_symbol : sympy.Symbol, optional
        Symbol representing time. Default: "t".

    Returns
    -------
    :
        Set of sympy Symbols representing parameters.
    """
    all_syms = set()
    for expr in rhs_dict.values():
        all_syms |= {s for s in expr.free_symbols
                     if s.name not in state_var_names and s != time_symbol}
    return all_syms


def build_param_map_from_rhs(
    rhs_dict: dict,
    state_var_names: set,
    time_symbol=sympy.Symbol("t"),
):
    """Canonicalize the simpler RHS set first and build the parameter map.

    Parameters
    ----------
    rhs_dict :
        Mapping of variable names to sympy expressions.
    state_var_names :
        Set of state variable names.
    time_symbol : sympy.Symbol, optional
        Symbol representing time. Default: ``t``.

    Returns
    -------
    :
        Canonicalized dict and parameter map dict.
    """

    def canonicalize_params(expr, state_var_names: set, param_map: dict, time_symbol=sympy.Symbol("t")):
        params = sorted(
            [s for s in expr.free_symbols
             if s.name not in state_var_names and s != time_symbol],
            key=str
        )

        for p in params:
            if p not in param_map:
                param_map[p] = sympy.Symbol(f"p{len(param_map)}")

        return expr.subs(param_map), param_map

    param_map = {}
    canonical = {}
    for var, expr in rhs_dict.items():
        canonical[var], param_map = canonicalize_params(expr, state_var_names, param_map, time_symbol)

    return canonical, param_map


def align_param_map(
    rhs_dict: dict,
    existing_param_map: dict,
    existing_canonical: dict,
    state_var_names: set,
    time_symbol=sympy.Symbol("t"),
):
    """Align parameter mapping between two sets of RHS expressions.

    Parameters
    ----------
    rhs_dict :
        Mapping of variable names to sympy expressions to align.
    existing_param_map :
        Parameter map from the reference model.
    existing_canonical :
        Canonicalized reference model.
    state_var_names :
        Set of state variable names.
    time_symbol : sympy.Symbol, optional
        Symbol representing time. Default: ``t``.

    Returns
    -------
    :
        Canonicalized dict and parameter map dict.
    """
    param_map = {}
    remaining_rhs = {}
    for var, expr in rhs_dict.items():
        if var not in existing_canonical:
            continue
        terms2 = sympy.Add.make_args(expr)
        flag = True
        for term2 in terms2:
            term2_params = [
                s for s in term2.free_symbols
                if s.name not in state_var_names and s != time_symbol
            ]
            if not term2_params:
                continue
            if all(p in param_map or p in existing_param_map for p in term2_params):
                for p in term2_params:
                    if p in existing_param_map and p not in param_map:
                        param_map[p] = existing_param_map[p]
                continue
            else:
                flag = False
        if not flag:
            remaining_rhs[var] = expr

    for var, expr in remaining_rhs.items():
        if var not in existing_canonical:
            continue
        numer, denom = fraction(expr)
        terms2 = [t / denom for t in expand(numer).as_ordered_terms()]
        numer, denom = fraction(existing_canonical[var])
        terms1_canonical = [t / denom for t in expand(numer).as_ordered_terms()]
        for term2 in terms2:
            partial_sub = {**existing_param_map, **param_map}
            term2_partial = term2.subs(partial_sub)
            for term1 in terms1_canonical:
                ratio = sympy.simplify(term2_partial / term1)
                ratio_syms = ratio.free_symbols
                unresolved = [
                    s
                    for s in ratio_syms
                    if s not in existing_param_map.values()
                    and not str(s).startswith("p")
                ]
                placeholders_in_ratio = [
                    s for s in ratio_syms if str(s).startswith("p")
                ]
                if (
                    len(unresolved) == 1
                    and len(placeholders_in_ratio) == 1
                    and ratio == unresolved[0] / placeholders_in_ratio[0]
                ):
                    if placeholders_in_ratio[0] not in param_map.values():
                        param_map[unresolved[0]] = placeholders_in_ratio[0]
                        break

    all_params = get_params(rhs_dict, state_var_names, time_symbol)
    used_placeholders = set(existing_param_map.values()) | set(param_map.values())
    for p in sorted(all_params, key=str):
        if p not in param_map and p not in existing_param_map:
            i = 0
            while sympy.Symbol(f"p{i}") in used_placeholders:
                i += 1
            new_ph = sympy.Symbol(f"p{i}")
            param_map[p] = new_ph
            used_placeholders.add(new_ph)
    canonical = {
        var: expr.subs({**existing_param_map, **param_map})
        for var, expr in rhs_dict.items()
    }
    return canonical, param_map


# ----------------------------------
# FULL PIPELINE
# ----------------------------------

def compare_models(ode_str1: str, ode_str2: str) -> dict:
    """Compare two ODE models at compartment, term-set, and tree-edit levels.

    Parameters
    ----------
    ode_str1 :
        ODE string for model 1.
    ode_str2 :
        ODE string for model 2.

    Returns
    -------
    :
        Results for each comparison layer.
    """
    rhs1 = parse_ode_string(ode_str1)
    rhs2 = parse_ode_string(ode_str2)

    state_vars = set(rhs1.keys()) | set(rhs2.keys())
    params1 = get_params(rhs1, state_vars)
    params2 = get_params(rhs2, state_vars)
    result = compartment_jaccard(rhs1, rhs2)
    rhs2_aligned = result.get("rhs2_aligned", rhs2)
    if len(params1) <= len(params2):
        canon1, param_map = build_param_map_from_rhs(rhs1, state_vars)
        canon2, _ = align_param_map(rhs2_aligned, param_map, canon1, state_vars)
    else:
        canon2, param_map = build_param_map_from_rhs(rhs2_aligned, state_vars)
        canon1, _ = align_param_map(rhs1, param_map, canon2, state_vars)
    return {
        "compartment_jaccard": result,
        "term_jaccard": term_set_jaccard(canon1, canon2),
        "ted": tree_edit_distance(canon1, canon2),
    }
