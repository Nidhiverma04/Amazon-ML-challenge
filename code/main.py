"""Convenience wrapper that runs all three pipeline stages back to back:

    1. generate_pairs.py  -- load training data, block, stream pairs to disk
    2. train.py            -- train on the saved pairs, pick a threshold
    3. predict.py           -- score the test set, write submission files

Each stage still runs as its own subprocess, so memory from one stage
(the blocking index in particular) is fully released before the next
stage starts -- this is the same reason the three stages exist as
separate commands in the first place. If stage 1 was the one crashing
your machine, running it on its own (`python generate_pairs.py`) is
easier to watch/retry than this all-in-one wrapper.

Usage:
    python main.py --train-dir dataset/train --test-dir dataset/test --output-dir output

Equivalent to running, in order:
    python generate_pairs.py --train-dir dataset/train --output-dir output
    python train.py --output-dir output
    python predict.py --test-dir dataset/test --output-dir output
"""
import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Run the full Business Entity Resolution pipeline.")
    parser.add_argument("--train-dir", default="dataset/train")
    parser.add_argument("--test-dir", default="dataset/test")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--flush-every", type=int, default=None)
    return parser.parse_args()


def run(cmd):
    print(f"\n$ {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


def main():
    args = parse_args()
    here = Path(__file__).parent

    generate_cmd = [
        sys.executable, str(here / "generate_pairs.py"),
        "--train-dir", args.train_dir,
        "--output-dir", args.output_dir,
    ]
    if args.flush_every is not None:
        generate_cmd += ["--flush-every", str(args.flush_every)]
    run(generate_cmd)

    train_cmd = [sys.executable, str(here / "train.py"), "--output-dir", args.output_dir]
    if args.threshold is not None:
        train_cmd += ["--threshold", str(args.threshold)]
    run(train_cmd)

    predict_cmd = [
        sys.executable, str(here / "predict.py"),
        "--test-dir", args.test_dir,
        "--output-dir", args.output_dir,
    ]
    run(predict_cmd)


if __name__ == "__main__":
    main()
