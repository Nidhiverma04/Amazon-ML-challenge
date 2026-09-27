from array import array
from collections import Counter, defaultdict

from .preprocessing import chargrams, token_set
from .progress import progress


def _keys(name, addr, country):
    """Blocking keys for one record. Same key scheme as the original code,
    just computed from raw strings instead of a pandas row + precomputed
    token-set columns."""
    keys = set()

    if name:
        keys.add("N|" + name)

    if addr:
        keys.add("A|" + addr)

    if country and name:
        for token in token_set(name):
            if len(token) >= 3:
                keys.add(f"CN|{country}|{token}")

    if country and addr:
        for token in token_set(addr):
            if len(token) >= 4:
                keys.add(f"CA|{country}|{token}")

    for gram in chargrams(name, 3):
        keys.add("NG|" + gram)

    for gram in chargrams(addr, 3):
        keys.add("AG|" + gram)

    return keys


class BlockIndex:
    """Multi-pass inverted index for scalable candidate generation.

    Memory-efficiency changes vs. the original implementation, which matter
    a lot once this runs over several million combined Source-2/Source-3
    rows:

    1. Two-pass build, no `row_keys` buffer. The original code computed the
       key-set for every single row up front and kept ALL of those sets
       resident in a `row_keys` list simultaneously (on top of the final
       index) just to know each key's total posting-list length before
       pruning. Here we do the same two logical passes -- one to count key
       frequencies, one to populate postings -- but recompute the (cheap)
       key-set for each row on the fly in both passes instead of storing
       millions of sets in between. Peak memory is the counts table plus
       the final index, not both of those plus every row's key-set.
    2. `array('i', ...)` postings instead of `list`. A Python `list` of
       row-index ints stores each int as its own ~28-byte object plus an
       8-byte pointer; a C-style `array('i')` packs them as raw 4-byte
       ints. For hundreds of millions of postings entries (this scale
       easily produces that many) this is roughly an 8x reduction for the
       index itself.
    3. Plain lists (via `RecordTable`) instead of a pandas DataFrame, so
       `.iterrows()` (slow, allocates a Series per row) is never used.
    """

    def __init__(self, table, max_posting=300, topk=80, desc="Blocking index"):
        self.table = table
        self.max_posting = max_posting
        self.topk = topk
        self.index = defaultdict(lambda: array("i"))
        self._build(desc)

    def _build(self, desc):
        n = len(self.table)
        names, addrs, countries = self.table.names, self.table.addrs, self.table.countries

        counts = Counter()
        for i in progress(range(n), total=n, desc=f"{desc}: counting keys", unit="rows"):
            counts.update(_keys(names[i], addrs[i], countries[i]))

        max_posting = self.max_posting
        index = self.index
        for i in progress(range(n), total=n, desc=f"{desc}: building postings", unit="rows"):
            for key in _keys(names[i], addrs[i], countries[i]):
                if counts[key] <= max_posting:
                    index[key].append(i)

        # `counts` (one entry per distinct key) can be freed once postings
        # are built; only `index` (postings for keys that survived pruning)
        # is needed afterwards.
        del counts

    def query(self, name, addr, country):
        votes = Counter()

        for key in _keys(name, addr, country):
            postings = self.index.get(key)
            if postings:
                for row_index in postings:
                    votes[row_index] += 1

        return [row_index for row_index, _ in votes.most_common(self.topk)]
