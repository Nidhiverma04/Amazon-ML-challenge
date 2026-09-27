"""Thin wrapper around tqdm used everywhere a progress bar is shown.

Every long-running loop in this pipeline (index building, pair generation,
model training, inference) goes through `progress()` so there is exactly one
place that decides how progress bars look and what happens if `tqdm` isn't
installed. `pip install -r requirements.txt` installs tqdm, but if it's
somehow missing we fall back to a plain iterator with periodic prints
instead of crashing.
"""

try:
    from tqdm import tqdm as _tqdm
    HAVE_TQDM = True
except ImportError:  # pragma: no cover - exercised only without tqdm
    HAVE_TQDM = False


class _FallbackBar:
    """Minimal drop-in replacement for tqdm(...) when tqdm isn't installed."""

    def __init__(self, iterable=None, total=None, desc=None, unit="it",
                 print_every=50000):
        self.iterable = iterable
        self.total = total
        self.desc = desc or ""
        self.unit = unit
        self.print_every = print_every
        self.n = 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.update(1)
        self._final_print()

    def update(self, amount=1):
        self.n += amount
        if self.n % self.print_every == 0:
            self._print()

    def set_postfix(self, **kwargs):
        self._postfix = kwargs

    def _print(self):
        suffix = f" / {self.total}" if self.total else ""
        print(f"{self.desc}: {self.n}{suffix} {self.unit}")

    def _final_print(self):
        self._print()

    def close(self):
        self._final_print()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()


def progress(iterable=None, total=None, desc=None, unit="it"):
    """Return a tqdm progress bar, or a lightweight fallback without tqdm."""
    if HAVE_TQDM:
        return _tqdm(iterable, total=total, desc=desc, unit=unit)
    return _FallbackBar(iterable, total=total, desc=desc, unit=unit)
