"""
Data loaders. Each returns a TIDY long DataFrame with columns:

    iso3 | indicator | year | value

and caches its result to data/raw/<source>.parquet so reruns don't hit the API.

Design note: this is the seam where a prior project's loader utilities
(e.g. an existing OECD data-loader / spread feed) would plug in. The public
contract each loader must satisfy is the tidy 4-column frame above.
"""
from __future__ import annotations

import pandas as pd

from .. import config

TIDY_COLS = ["iso3", "indicator", "year", "value"]


def cache_path(source: str):
    return config.RAW_DIR / f"{source}.parquet"


def save_cache(df: pd.DataFrame, source: str) -> None:
    df.to_parquet(cache_path(source), index=False)


def load_cache(source: str) -> pd.DataFrame | None:
    p = cache_path(source)
    if p.exists():
        return pd.read_parquet(p)
    return None
