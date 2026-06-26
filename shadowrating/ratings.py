"""
Ratings: letter <-> numeric notch mapping, and loading the target CSV.

Numeric scale runs AAA/Aaa = 21 down to D/RD/C = 1, matching the build spec.
The investment-grade / high-yield boundary sits between notch 12 (BBB-/Baa3)
and notch 11 (BB+/Ba1) -- crossing it is the single most consequential event,
so it is flagged explicitly via `ig_flag`.
"""
from __future__ import annotations

import pandas as pd

from . import config

IG_BOUNDARY_NOTCH = 12  # >= 12 is investment grade

# S&P and Fitch share one alphabetic scale.
SP_FITCH_SCALE: dict[str, int] = {
    "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18,
    "A+": 17, "A": 16, "A-": 15,
    "BBB+": 14, "BBB": 13, "BBB-": 12,
    "BB+": 11, "BB": 10, "BB-": 9,
    "B+": 8, "B": 7, "B-": 6,
    "CCC+": 5, "CCC": 4, "CCC-": 3,
    "CC": 2,
    "SD": 1, "D": 1, "RD": 1,
}

# Moody's scale, aligned notch-for-notch with the above.
MOODYS_SCALE: dict[str, int] = {
    "AAA": 21, "AA1": 20, "AA2": 19, "AA3": 18,
    "A1": 17, "A2": 16, "A3": 15,
    "BAA1": 14, "BAA2": 13, "BAA3": 12,
    "BA1": 11, "BA2": 10, "BA3": 9,
    "B1": 8, "B2": 7, "B3": 6,
    "CAA1": 5, "CAA2": 4, "CAA3": 3,
    "CA": 2,
    "C": 1,
}


def _normalize(raw: str) -> str:
    """Uppercase, strip whitespace, and turn the Unicode minus into ASCII '-'."""
    if raw is None:
        return ""
    return str(raw).strip().upper().replace("\u2212", "-").replace(" ", "")


def sp_fitch_to_notch(rating: str) -> int | None:
    return SP_FITCH_SCALE.get(_normalize(rating))


def moodys_to_notch(rating: str) -> int | None:
    return MOODYS_SCALE.get(_normalize(rating))


def notch_to_sp_letter(notch: float | None) -> str | None:
    """Inverse map (rounded) back to an S&P-style letter, for display."""
    if notch is None or pd.isna(notch):
        return None
    inv = {v: k for k, v in {
        "AAA": 21, "AA+": 20, "AA": 19, "AA-": 18, "A+": 17, "A": 16, "A-": 15,
        "BBB+": 14, "BBB": 13, "BBB-": 12, "BB+": 11, "BB": 10, "BB-": 9,
        "B+": 8, "B": 7, "B-": 6, "CCC+": 5, "CCC": 4, "CCC-": 3, "CC": 2, "D": 1,
    }.items()}
    return inv.get(int(round(notch)))


# Six broad rating bands, matching the major letter families (and the
# `bucket` labels already used for country sampling in config.py). Phase 3's
# ordered logit predicts bands rather than the full 21-notch scale -- with
# N~42, fitting 20 thresholds is not honest; 5 thresholds across 6 ordered
# bands is.
RATING_BANDS: list[tuple[str, int, int]] = [
    ("CCC-D", 1, 5),
    ("B", 6, 8),
    ("BB", 9, 11),
    ("BBB", 12, 14),
    ("A", 15, 17),
    ("AAA/AA", 18, 21),
]


def notch_to_band(notch: float | None) -> int | None:
    """Numeric notch -> band index 0 (CCC-D) .. 5 (AAA/AA)."""
    if notch is None or pd.isna(notch):
        return None
    n = round(float(notch))
    for i, (_, lo, hi) in enumerate(RATING_BANDS):
        if lo <= n <= hi:
            return i
    return 0 if n < 1 else len(RATING_BANDS) - 1


def band_label(band: float | None) -> str | None:
    if band is None or pd.isna(band):
        return None
    i = int(round(float(band)))
    i = max(0, min(len(RATING_BANDS) - 1, i))
    return RATING_BANDS[i][0]


def load_ratings(path=config.RATINGS_CSV) -> pd.DataFrame:
    """
    Load the target CSV and attach numeric notches + a consensus.

    Returns one row per country with:
      iso3, name, sp, fitch, moodys,
      sp_notch, fitch_notch, moodys_notch,
      consensus_notch (mean of available, rounded), consensus_letter,
      ig_flag (True = investment grade), n_agencies, split (max-min notch spread).
    """
    df = pd.read_csv(path, comment="#")

    df["sp_notch"] = df["sp"].map(sp_fitch_to_notch)
    df["fitch_notch"] = df["fitch"].map(sp_fitch_to_notch)
    df["moodys_notch"] = df["moodys"].map(moodys_to_notch)

    notch_cols = ["sp_notch", "fitch_notch", "moodys_notch"]

    # Warn loudly about any rating string we failed to parse.
    for col, src in zip(notch_cols, ["sp", "fitch", "moodys"]):
        bad = df.loc[df[col].isna() & df[src].notna(), ["iso3", src]]
        if not bad.empty:
            print(f"[ratings] WARNING: unparsed {src} ratings:\n{bad.to_string(index=False)}")

    df["consensus_notch"] = df[notch_cols].mean(axis=1, skipna=True).round().astype("Int64")
    df["consensus_letter"] = df["consensus_notch"].map(notch_to_sp_letter)
    df["n_agencies"] = df[notch_cols].notna().sum(axis=1)
    df["split"] = (df[notch_cols].max(axis=1) - df[notch_cols].min(axis=1)).astype("Int64")
    df["ig_flag"] = df["consensus_notch"] >= IG_BOUNDARY_NOTCH
    df["consensus_band"] = df["consensus_notch"].map(notch_to_band)

    return df


if __name__ == "__main__":
    out = load_ratings()
    cols = ["iso3", "name", "sp", "fitch", "moodys",
            "consensus_notch", "consensus_letter", "ig_flag", "split"]
    print(out[cols].to_string(index=False))
