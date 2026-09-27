from dataclasses import dataclass
import os

@dataclass(frozen=True)
class PipelineConfig:
    max_posting: int = 300
    topk_candidates: int = 80
    max_neg_per_s1: int = 20
    default_threshold: float = 0.70
    max_matches_per_s1: int = 20
    random_state: int = 42
    n_estimators: int = 500
    max_depth: int = 6
    learning_rate: float = 0.05
    subsample: float = 0.85
    colsample_bytree: float = 0.90
    min_child_weight: int = 3
    reg_lambda: float = 3.0
    n_jobs: int = max(1, (os.cpu_count() or 2) - 1)

    # How many pair rows to buffer in memory before flushing to disk while
    # generating training/validation pairs. This is the main memory-vs-speed
    # knob for the `generate_pairs` stage: memory use stays roughly constant
    # (~this many rows x ~15 floats) no matter how many millions of pairs
    # are generated in total, because we never hold the full pair set in
    # RAM at once. Lower it (even to 1) for a literal "one pair at a time"
    # save; raise it for fewer, larger disk writes and higher throughput.
    pair_flush_every: int = 5000
