"""Stage 1: load training data, build the blocking index, and stream all
training + validation candidate pairs to disk.

This is the memory-heavy, slow stage (it touches every Source-2/Source-3
row and every Source-1 entity), so it's kept as its own process/command:
once it exits, all of its memory -- the raw dataframes, the blocking
index, everything -- is released back to the OS before `train.py` ever
starts.

Usage:
    python generate_pairs.py --train-dir dataset/train --output-dir output

Outputs (all under --output-dir):
    train_pairs.tsv        labeled pairs to fit the model on (negatives
                            capped per Source-1 entity, same as the
                            original pipeline)
    validation_pairs.tsv   labeled pairs for threshold selection
                            (uncapped -- mirrors what inference will see)
    validation_truth.json  {source1_entity_id: [true match ids]} for every
                            validation entity, including ones with no
                            candidates at all
"""
import argparse
import gc
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupShuffleSplit

from src.config import PipelineConfig
from src.preprocessing import prepare_dataframe, RecordTable
from src.io_utils import load_source, load_ground_truth, save_truth_json, save_json
from src.training import parse_ground_truth
from src.pair_generation import build_combined_index, generate_pairs_to_tsv


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate training + validation candidate pairs and save them to disk."
    )
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--flush-every",
        type=int,
        default=None,
        help="Rows to buffer before each disk write (default: PipelineConfig.pair_flush_every). "
             "Lower = closer to 'save every pair immediately', at the cost of more, smaller "
             "disk writes. Higher = faster, more memory used for the buffer.",
    )
    return parser.parse_args()


def _load_table(path, desc):
    print(f"Loading {desc}...")
    df = prepare_dataframe(load_source(path))
    table = RecordTable.from_dataframe(df)
    del df
    gc.collect()
    print(f"  {len(table):,} rows")
    return table


def main():
    args = parse_args()
    config = PipelineConfig()
    if args.flush_every is not None:
        config = PipelineConfig(**{**config.__dict__, "pair_flush_every": args.flush_every})

    train_dir = Path(args.train_dir)
    output_dir = Path(args.output_dir)

    source1_table = _load_table(train_dir / "train_source1.tsv", "Source 1 (reference)")
    source2_table = _load_table(train_dir / "train_source2.tsv", "Source 2")
    source3_table = _load_table(train_dir / "train_source3.tsv", "Source 3")

    print("Building combined Source-2 + Source-3 candidate table...")
    candidate_table = RecordTable.concat([source2_table, source3_table])
    del source2_table, source3_table
    gc.collect()
    print(f"  {len(candidate_table):,} candidate rows total")

    print("Loading ground truth...")
    ground_truth = parse_ground_truth(load_ground_truth(train_dir / "train_ground_truth.tsv"))

    # Split by Source-1 entity so one entity cannot occur in both folds.
    # Source-1 entity_ids are distinct, so this is effectively a plain
    # 80/20 shuffle-split; GroupShuffleSplit is kept so behaviour matches
    # the original pipeline exactly if that ever changes.
    n = len(source1_table)
    groups = np.array(source1_table.ids)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=config.random_state)
    fit_indices, validation_indices = next(splitter.split(np.zeros(n), groups=groups))

    fit_table = source1_table.subset(fit_indices)
    validation_table = source1_table.subset(validation_indices)
    del source1_table
    gc.collect()

    print(f"Fit split: {len(fit_table):,} entities | Validation split: {len(validation_table):,} entities")

    print("Building blocking index over Source 2 + Source 3 (this is the memory-heavy step)...")
    index = build_combined_index(candidate_table, config, desc="Blocking index")

    print("Generating training pairs (negatives capped per entity)...")
    n_fit_entities, n_fit_pairs, n_fit_pos = generate_pairs_to_tsv(
        fit_table,
        candidate_table,
        index,
        ground_truth,
        config,
        output_dir / "train_pairs.tsv",
        cap_negatives=True,
        desc="Generating training pairs",
    )
    print(
        f"  {n_fit_pairs:,} training pairs written "
        f"({n_fit_pos:,} positive, {n_fit_pairs - n_fit_pos:,} negative)"
    )

    print("Generating validation pairs (uncapped, for threshold selection)...")
    validation_truth = {
        source1_id: ground_truth.get(source1_id, set())
        for source1_id in validation_table.ids
    }
    n_val_entities, n_val_pairs, n_val_pos = generate_pairs_to_tsv(
        validation_table,
        candidate_table,
        index,
        ground_truth,
        config,
        output_dir / "validation_pairs.tsv",
        cap_negatives=False,
        desc="Generating validation pairs",
    )
    print(
        f"  {n_val_pairs:,} validation pairs written "
        f"({n_val_pos:,} positive, {n_val_pairs - n_val_pos:,} negative)"
    )

    save_truth_json(validation_truth, output_dir / "validation_truth.json")
    save_json(
        {
            "n_fit_entities": n_fit_entities,
            "n_fit_pairs": n_fit_pairs,
            "n_fit_positive_pairs": n_fit_pos,
            "n_validation_entities": n_val_entities,
            "n_validation_pairs": n_val_pairs,
            "n_validation_positive_pairs": n_val_pos,
            "random_state": config.random_state,
        },
        output_dir / "pair_generation_summary.json",
    )

    print(f"Wrote: {output_dir / 'train_pairs.tsv'}")
    print(f"Wrote: {output_dir / 'validation_pairs.tsv'}")
    print(f"Wrote: {output_dir / 'validation_truth.json'}")
    print("Done. Next: python train.py --output-dir", args.output_dir)


if __name__ == "__main__":
    main()
