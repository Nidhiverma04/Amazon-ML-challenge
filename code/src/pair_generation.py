"""Streaming generation of labeled (source1, candidate) pairs.

This replaces the old `training.make_training_pairs`, which built the
entire pair set as Python lists (`X.append(...)`, `y.append(...)`) held
fully in RAM for the whole run. At multi-million-row scale that list of
lists is one of the two or three biggest memory users in the pipeline.

Here, pairs are computed one Source-1 entity at a time and written out in
small batches (`config.pair_flush_every` rows at a time) as soon as they're
ready, so peak memory for pair generation is bounded by that batch size --
not by the total number of pairs in the whole dataset. A live progress bar
(tqdm) reports entities processed and pairs written as it goes.
"""

import csv
from pathlib import Path

from .blocking import BlockIndex
from .features import pair_features, FEATURE_NAMES
from .progress import progress

PAIR_FILE_HEADER = ["source1_entity_id", "candidate_entity_id", "label"] + FEATURE_NAMES


def build_combined_index(candidate_table, config, desc="Blocking index"):
    """Build the Source-2 + Source-3 blocking index once, so it can be
    reused for both the training-pair pass and the validation-pair pass."""
    return BlockIndex(
        candidate_table,
        max_posting=config.max_posting,
        topk=config.topk_candidates,
        desc=desc,
    )


def generate_pairs_to_tsv(
    source1_table,
    candidate_table,
    index,
    truth,
    config,
    output_path,
    cap_negatives,
    desc="Generating pairs",
):
    """Stream labeled candidate pairs for every entity in `source1_table`
    to `output_path` (tab-separated), flushing every `config.pair_flush_every`
    rows.

    cap_negatives=True reproduces the original training-pair behaviour
    (at most `config.max_neg_per_s1` negatives kept per Source-1 entity, to
    keep the class balance and dataset size sane for fitting). Use
    cap_negatives=False for validation pairs, so threshold selection sees
    the same, uncapped candidate set that will be scored at inference time.

    Returns (n_entities, n_pairs_written, n_positive_pairs).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    entity_to_row = {eid: i for i, eid in enumerate(candidate_table.ids)}

    names, addrs, countries = (
        source1_table.names,
        source1_table.addrs,
        source1_table.countries,
    )
    cand_names, cand_addrs, cand_countries = (
        candidate_table.names,
        candidate_table.addrs,
        candidate_table.countries,
    )

    n_entities = len(source1_table)
    n_pairs = 0
    n_positive = 0
    buffer = []
    flush_every = max(1, config.pair_flush_every)

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(PAIR_FILE_HEADER)

        bar = progress(range(n_entities), total=n_entities, desc=desc, unit="entity")
        for i in bar:
            s1_id = source1_table.ids[i]
            name_a, addr_a, country_a = names[i], addrs[i], countries[i]

            candidates = index.query(name_a, addr_a, country_a)
            positives = truth.get(s1_id, set())

            # Guarantee that known training positives are included even if
            # blocking missed them, exactly as the original code did.
            for matched_id in positives:
                row_index = entity_to_row.get(matched_id)
                if row_index is not None:
                    candidates.append(row_index)

            candidates = list(dict.fromkeys(candidates))

            positive_rows, negative_rows = [], []
            for row_index in candidates:
                if candidate_table.ids[row_index] in positives:
                    positive_rows.append(row_index)
                else:
                    negative_rows.append(row_index)

            if cap_negatives and len(negative_rows) > config.max_neg_per_s1:
                negative_rows = sorted(negative_rows)[:config.max_neg_per_s1]

            for row_index in positive_rows + negative_rows:
                cand_id = candidate_table.ids[row_index]
                label = int(cand_id in positives)
                feats = pair_features(
                    name_a, addr_a, country_a,
                    cand_names[row_index], cand_addrs[row_index], cand_countries[row_index],
                )
                buffer.append([s1_id, cand_id, label] + feats)
                n_pairs += 1
                n_positive += label

            if len(buffer) >= flush_every:
                writer.writerows(buffer)
                buffer.clear()
                if hasattr(bar, "set_postfix"):
                    bar.set_postfix(pairs=n_pairs, positives=n_positive)

        if buffer:
            writer.writerows(buffer)
            buffer.clear()
        if hasattr(bar, "set_postfix"):
            bar.set_postfix(pairs=n_pairs, positives=n_positive)

    return n_entities, n_pairs, n_positive
