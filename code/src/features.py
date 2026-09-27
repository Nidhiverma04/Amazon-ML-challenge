import numpy as np
from rapidfuzz.fuzz import ratio, WRatio

from .preprocessing import compact, token_set

FEATURE_NAMES = [
    "country_eq",
    "name_exact",
    "addr_exact",
    "name_ratio",
    "name_wratio",
    "addr_ratio",
    "addr_wratio",
    "name_jaccard",
    "name_overlap",
    "addr_jaccard",
    "addr_overlap",
    "name_prefix",
    "addr_prefix",
    "name_len_diff",
    "addr_len_diff",
]

def jaccard(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def overlap(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))

def pair_features(name_a, addr_a, country_a, name_b, addr_b, country_b):
    """Feature vector for one (source1, candidate) pair, from plain strings.

    Token sets are computed here on demand (see `preprocessing.token_set`)
    rather than being read off precomputed DataFrame columns -- this keeps
    the caller free to hold only raw normalized strings for the full
    multi-million-row source tables instead of millions of extra `set`
    objects.
    """
    name_tok_a, name_tok_b = token_set(name_a), token_set(name_b)
    addr_tok_a, addr_tok_b = token_set(addr_a), token_set(addr_b)

    return [
        float(country_a == country_b and country_a != ""),
        float(name_a == name_b and name_a != ""),
        float(addr_a == addr_b and addr_a != ""),
        ratio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        WRatio(name_a, name_b) / 100.0 if name_a and name_b else 0.0,
        ratio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        WRatio(addr_a, addr_b) / 100.0 if addr_a and addr_b else 0.0,
        jaccard(name_tok_a, name_tok_b),
        overlap(name_tok_a, name_tok_b),
        jaccard(addr_tok_a, addr_tok_b),
        overlap(addr_tok_a, addr_tok_b),
        float(
            compact(name_a)[:6] == compact(name_b)[:6]
            and bool(name_a and name_b)
        ),
        float(
            compact(addr_a)[:8] == compact(addr_b)[:8]
            and bool(addr_a and addr_b)
        ),
        abs(len(name_a) - len(name_b)),
        abs(len(addr_a) - len(addr_b)),
    ]

def build_feature_matrix(source_row, candidate_table, candidate_indices):
    """Feature matrix for one source1 record against several candidates.

    Used by inference to score every candidate for a Source-1 record with a
    single `model.predict_proba(...)` call instead of one call per
    candidate (which is what made the original `inference.py` slow: an
    XGBoost call has fixed per-call overhead that dominates when you make
    tens of candidate-sized calls per Source-1 row).
    """
    name_a, addr_a, country_a = source_row
    names, addrs, countries = (
        candidate_table.names,
        candidate_table.addrs,
        candidate_table.countries,
    )

    rows = [
        pair_features(
            name_a, addr_a, country_a,
            names[i], addrs[i], countries[i],
        )
        for i in candidate_indices
    ]

    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)
