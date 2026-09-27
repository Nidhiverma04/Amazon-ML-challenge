import numpy as np

from .features import FEATURE_NAMES
from .config import PipelineConfig

def f05_for_entity(predicted, truth):
    predicted = set(predicted)
    truth = set(truth)

    if not predicted and not truth:
        return 1.0
    if not predicted or not truth:
        return 0.0

    precision = len(predicted & truth) / len(predicted)
    recall = len(predicted & truth) / len(truth)

    denominator = 0.25 * precision + recall
    return (1.25 * precision * recall) / denominator if denominator else 0.0

def score_macro_f05(predictions, truth):
    scores = []

    for source1_id, predicted in predictions.items():
        scores.append(
            f05_for_entity(predicted, truth.get(source1_id, set()))
        )

    return float(np.mean(scores)) if scores else 0.0

def choose_threshold_from_pairs(
    model,
    validation_pairs_df,
    validation_truth,
    config: PipelineConfig,
    thresholds=None,
):
    """Pick the probability threshold that maximizes macro F0.5 on a
    validation candidate-pairs table (as produced by
    `pair_generation.generate_pairs_to_tsv(..., cap_negatives=False)`).

    Unlike the original `choose_threshold`, which called
    `model.predict_proba` once per individual candidate row inside a Python
    loop (extremely slow -- each call pays XGBoost's fixed per-call
    overhead), this scores every candidate in the validation set with a
    single batched `predict_proba` call.

    `validation_truth` must cover every Source-1 entity in the validation
    split, including ones with an empty match list -- entities that got
    zero blocking candidates (and so have no rows in
    `validation_pairs_df`) still need to be counted as correct empty
    predictions when their true match list is also empty.
    """
    if len(validation_pairs_df) == 0:
        probabilities = np.empty(0, dtype=np.float32)
    else:
        X = validation_pairs_df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        probabilities = model.predict_proba(X)[:, 1]

    source1_ids = validation_pairs_df["source1_entity_id"].to_numpy()
    candidate_ids = validation_pairs_df["candidate_entity_id"].to_numpy()

    if thresholds is None:
        thresholds = np.arange(0.50, 0.96, 0.025)

    best_score = -1.0
    best_threshold = config.default_threshold
    # All validation Source-1 entities, including ones with zero candidates.
    all_ids = list(validation_truth.keys())

    for threshold in thresholds:
        keep = probabilities >= threshold
        predictions = {source1_id: [] for source1_id in all_ids}

        for source1_id, candidate_id in zip(source1_ids[keep], candidate_ids[keep]):
            predictions.setdefault(source1_id, []).append(candidate_id)

        score = score_macro_f05(predictions, validation_truth)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_score, best_threshold
