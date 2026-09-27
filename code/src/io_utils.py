import json
from pathlib import Path
import pandas as pd

def load_source(path):
    return pd.read_csv(path, sep="\t")

def load_ground_truth(path):
    return pd.read_csv(
        path,
        sep="\t",
        keep_default_na=False,
    )

def write_id_lists(rows, path, column_name):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(f"source1_entity_id\t{column_name}\n")

        for source1_id, entity_ids in rows:
            file.write(
                f"{source1_id}\t{','.join(entity_ids)}\n"
            )

def save_truth_json(truth, path):
    """Persist a {source1_entity_id: {matched_ids...}} map as JSON so a
    later pipeline stage (running in its own process, without the raw
    ground-truth TSV or the train/validation split reloaded) can still
    score against it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump({key: sorted(value) for key, value in truth.items()}, file)

def load_truth_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    return {key: set(value) for key, value in data.items()}

def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)

def load_json(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)
