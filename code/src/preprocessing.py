import re
import unicodedata
from dataclasses import dataclass

import pandas as pd

LEGAL = {
    "corporation": "", "corp": "", "incorporated": "", "inc": "",
    "limited": "", "ltd": "", "private": "", "pvt": "", "llc": "",
    "llp": "", "plc": "", "company": "", "co": "", "gmbh": "",
    "sarl": "", "sas": ""
}

ABBR = {
    "road": "rd", "street": "st", "avenue": "ave", "av": "ave",
    "boulevard": "blvd", "drive": "dr", "lane": "ln", "highway": "hwy",
    "apartment": "apt", "building": "bldg", "floor": "fl",
    "centre": "center"
}

def normalize(value):
    if pd.isna(value):
        return ""

    value = (
        unicodedata.normalize("NFKD", str(value))
        .encode("ascii", "ignore")
        .decode()
        .lower()
    )
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)

    tokens = []
    for token in value.split():
        token = ABBR.get(token, token)
        if token in LEGAL and LEGAL[token] == "":
            continue
        tokens.append(token)

    return " ".join(tokens)

def compact(value):
    return re.sub(r"[^a-z0-9]", "", value)

def token_set(value):
    """Recomputed on demand rather than stored per-row.

    Storing a Python `set` object per row for millions of rows (as the
    original code did via `_name_tok`/`_addr_tok` DataFrame columns) is one
    of the biggest memory sinks at multi-million-row scale: each `set`
    carries substantial per-object overhead on top of its string elements,
    multiplied by every row in every source. `value` here is already the
    normalized string, so `.split()` is cheap -- recomputing it the handful
    of times it's needed is far lighter than keeping ~10M+ set objects
    resident in RAM for the whole run.
    """
    return set(value.split())

def chargrams(value, n=3):
    value = f"  {compact(value)}  "
    if len(value) <= n:
        return {value}
    return {value[i:i + n] for i in range(len(value) - n + 1)}

def prepare_dataframe(df):
    """Normalize the raw columns in place (adds `_name`/`_addr`/`_country`).

    Unlike the original version, this no longer materializes `_name_tok`/
    `_addr_tok` set columns -- see `token_set()` for why. Callers that need
    token sets should compute them from `_name`/`_addr` when needed, or
    convert to a `RecordTable` (below) and use `token_set()` inline.
    """
    df = df.copy()

    for column in ("business_name", "business_address", "country"):
        df[column] = df[column].fillna("").astype(str)

    df["_name"] = df["business_name"].map(normalize)
    df["_addr"] = df["business_address"].map(normalize)
    df["_country"] = df["country"].map(normalize)

    return df

@dataclass
class RecordTable:
    """Plain-list columnar view of a prepared source dataframe.

    Repeatedly pulling rows out of a pandas DataFrame with `.iterrows()` or
    `.iloc[i]` is both slow (each access builds a new pandas Series) and
    memory-heavy at multi-million-row scale. A RecordTable holds the same
    data as four flat Python lists, so `table.ids[i]` / `table.names[i]`
    etc. are plain O(1) list lookups with no per-access object churn.

    Build one with `RecordTable.from_dataframe(df)` right after
    `prepare_dataframe`, then drop the DataFrame (`del df`) to free the
    pandas overhead once you no longer need it.
    """

    ids: list
    names: list
    addrs: list
    countries: list

    def __len__(self):
        return len(self.ids)

    @classmethod
    def from_dataframe(cls, df):
        return cls(
            ids=df["entity_id"].tolist(),
            names=df["_name"].tolist(),
            addrs=df["_addr"].tolist(),
            countries=df["_country"].tolist(),
        )

    def subset(self, indices):
        return RecordTable(
            ids=[self.ids[i] for i in indices],
            names=[self.names[i] for i in indices],
            addrs=[self.addrs[i] for i in indices],
            countries=[self.countries[i] for i in indices],
        )

    @classmethod
    def concat(cls, tables):
        ids, names, addrs, countries = [], [], [], []
        for table in tables:
            ids.extend(table.ids)
            names.extend(table.names)
            addrs.extend(table.addrs)
            countries.extend(table.countries)
        return cls(ids=ids, names=names, addrs=addrs, countries=countries)
