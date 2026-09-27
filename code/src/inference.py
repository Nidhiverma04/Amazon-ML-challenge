from .blocking import BlockIndex
from .features import build_feature_matrix
from .config import PipelineConfig
from .progress import progress

def infer(
    source1_table,
    candidate_table,
    model,
    threshold,
    config: PipelineConfig,
):
    """Score every Source-1 entity against its blocked candidates.

    Scores every candidate for a given Source-1 entity with one batched
    `model.predict_proba` call (via `build_feature_matrix`) instead of one
    call per candidate -- the original code called `predict_proba` inside
    the per-candidate loop, which is far slower at this scale.
    """
    index = BlockIndex(
        candidate_table,
        max_posting=config.max_posting,
        topk=config.topk_candidates,
        desc="Inference blocking index",
    )

    matches = []
    candidates_out = []

    total = len(source1_table)
    names, addrs, countries = (
        source1_table.names,
        source1_table.addrs,
        source1_table.countries,
    )

    for i in progress(range(total), total=total, desc="Scoring candidates", unit="entity"):
        s1_id = source1_table.ids[i]
        name_a, addr_a, country_a = names[i], addrs[i], countries[i]

        row_indices = index.query(name_a, addr_a, country_a)
        candidate_ids = [candidate_table.ids[j] for j in row_indices]

        if row_indices:
            features = build_feature_matrix(
                (name_a, addr_a, country_a), candidate_table, row_indices
            )
            probabilities = model.predict_proba(features)[:, 1]
        else:
            probabilities = []

        scored = list(zip(candidate_ids, probabilities))

        # IMPORTANT:
        # This is the exact set that the final matching model scored.
        candidate_ids_dedup = list(dict.fromkeys(candidate_ids))

        matched_ids = [
            entity_id
            for entity_id, probability in sorted(
                scored,
                key=lambda item: item[1],
                reverse=True,
            )
            if probability >= threshold
        ][:config.max_matches_per_s1]

        candidates_out.append((s1_id, candidate_ids_dedup))
        matches.append((s1_id, matched_ids))

    return matches, candidates_out
