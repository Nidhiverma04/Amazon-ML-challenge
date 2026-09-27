"""Stage 3: score the test set with the trained model and write submission files.

Usage:
    python predict.py --test-dir dataset/test --output-dir output
"""
import argparse
from pathlib import Path

from xgboost import XGBClassifier

from src.config import PipelineConfig
from src.preprocessing import prepare_dataframe, RecordTable
from src.io_utils import load_source, write_id_lists, load_json
from src.inference import infer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained model over the test set and write matching_results.tsv / candidate_pairs.tsv."
    )
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the threshold saved by train.py.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = PipelineConfig()
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)

    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = load_json(output_dir / "threshold.json")["threshold"]
    print(f"Using threshold: {threshold:.3f}")

    print("Loading model...")
    model = XGBClassifier()
    model.load_model(output_dir / "model.json")

    print("Loading test data...")
    test_source1 = RecordTable.from_dataframe(
        prepare_dataframe(load_source(test_dir / "test_source1.tsv"))
    )
    test_source2 = RecordTable.from_dataframe(
        prepare_dataframe(load_source(test_dir / "test_source2.tsv"))
    )
    test_source3 = RecordTable.from_dataframe(
        prepare_dataframe(load_source(test_dir / "test_source3.tsv"))
    )
    test_candidates = RecordTable.concat([test_source2, test_source3])
    del test_source2, test_source3

    print("Running test inference...")
    matches, candidates = infer(
        test_source1,
        test_candidates,
        model,
        threshold,
        config,
    )

    write_id_lists(matches, output_dir / "matching_results.tsv", "matched_entity_ids")
    write_id_lists(candidates, output_dir / "candidate_pairs.tsv", "candidate_entity_ids")

    print(f"Wrote: {output_dir / 'matching_results.tsv'}")
    print(f"Wrote: {output_dir / 'candidate_pairs.tsv'}")


if __name__ == "__main__":
    main()
