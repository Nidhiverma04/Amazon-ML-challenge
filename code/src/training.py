import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost.callback import TrainingCallback

from .features import FEATURE_NAMES
from .config import PipelineConfig
from .progress import HAVE_TQDM, progress

def parse_ground_truth(gt):
    truth = {}

    for _, row in gt.iterrows():
        value = str(row["matched_entity_ids"])
        truth[row["source1_entity_id"]] = {
            x.strip()
            for x in value.split(",")
            if x.strip() and x.strip().lower() != "nan"
        }

    return truth

_PAIR_DTYPES = {
    "source1_entity_id": "string",
    "candidate_entity_id": "string",
    "label": "int8",
    **{name: "float32" for name in FEATURE_NAMES},
}

def load_pairs(path):
    """Load a pairs TSV written by `pair_generation.generate_pairs_to_tsv`.

    This is a plain `read_csv` -- the pairs file is already the small,
    bounded artifact produced by blocking + capping, not the raw
    multi-million-row source tables, so loading it whole in a fresh process
    is safe even on the same 24GB machine that couldn't hold the blocking
    index and the raw pair lists in memory at the same time.
    """
    return pd.read_csv(path, sep="\t", dtype=_PAIR_DTYPES, keep_default_na=False)

def pairs_to_xy(pairs_df):
    X = pairs_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
    y = pairs_df["label"].to_numpy(dtype=np.int8)
    groups = pairs_df["source1_entity_id"].to_numpy()
    return X, y, groups

class _TqdmBoostCallback(TrainingCallback):
    """xgboost TrainingCallback that drives a tqdm bar, one tick per tree."""

    def __init__(self, total):
        self._bar = progress(total=total, desc="Training XGBoost", unit="tree")

    def before_training(self, model):
        return model

    def after_iteration(self, model, epoch, evals_log):
        self._bar.update(1)
        return False  # False == "don't stop early"

    def after_training(self, model):
        self._bar.close()
        return model

def train_model(X, y, config: PipelineConfig):
    model = XGBClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        min_child_weight=config.min_child_weight,
        reg_lambda=config.reg_lambda,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=config.n_jobs,
        random_state=config.random_state,
        callbacks=[_TqdmBoostCallback(config.n_estimators)],
    )

    if not HAVE_TQDM:
        print(
            "tqdm not installed -- training progress will print periodically "
            "instead of a live bar. `pip install tqdm` for the live version."
        )

    model.fit(X, y)
    return model
