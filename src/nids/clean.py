"""Leak-free cleaning (PREREG §3): non-finite rows -> leak columns -> label conflicts -> duplicates.
Adds a `group` column (PREREG §4) computed from identifiers before they are dropped.
"""
import numpy as np
import pandas as pd

LABELS = ["y", "family", "src", "group"]

LEAK_COLS = {
    "unsw_official": ["sttl", "dttl", "ct_state_ttl", "label"],
    "unsw_raw": ["srcip", "sport", "dstip", "dsport", "stime", "ltime", "sttl", "dttl", "ct_state_ttl", "label"],
    "bot_iot": ["pkSeqID", "saddr", "sport", "daddr", "dport", "seq", "attack", "subcategory"],
    "ciciot_paper": ["Number"],
    "ciciot_csv": ["Number"],
    "ciciot_merged": ["Number"],
}


def group_ids(df, dataset):
    """Split groups: tenth of a capture file or 10-minute stime window. None = no group id (stratified random split).
    BoT-IoT has none: its pkSeqID follows class order, not time (PREREG §9)."""
    if dataset == "ciciot_csv":
        # 10 contiguous blocks per capture file: BruteForce has 1 file and Spoofing 3, so whole files are too coarse
        codes = pd.factorize(df["src"])[0]
        pos = pd.Series(codes).groupby(codes).cumcount().to_numpy()
        return pd.Series(codes * 10 + pos * 10 // np.bincount(codes)[codes], index=df.index)
    if dataset == "unsw_raw":
        return df["stime"] // 600
    return None


def feature_columns(df, dataset=None):
    """Feature columns; with `dataset`, also excluding that dataset's leak columns."""
    drop = set(LABELS) | set(LEAK_COLS[dataset] if dataset else ())
    return [c for c in df.columns if c not in drop]


def row_hash(df, cols):
    """64-bit hash per row over `cols`, built column by column so a 46M-row frame is never copied."""
    h = np.zeros(len(df), dtype=np.uint64)
    for c in cols:
        h *= np.uint64(1_000_003)
        h ^= pd.util.hash_array(df[c].to_numpy())
    return h


def clean(df, dataset):
    """Return (cleaned frame, report). Report counts are rows removed at each step, overall and per class.
    Works on masks and copies the frame once at the end (the CICIoT frames are ~9 GB)."""
    y = df["y"].astype(str).to_numpy()
    groups = group_ids(df, dataset)
    leak = set(LEAK_COLS[dataset])
    feats = feature_columns(df, dataset)

    finite = np.ones(len(df), dtype=bool)
    for c in feats:
        v = df[c]
        finite &= np.isfinite(v.to_numpy(dtype="float64", na_value=np.nan)) if pd.api.types.is_numeric_dtype(v) \
            else v.notna().to_numpy()

    h = row_hash(df, feats)
    fin = np.flatnonzero(finite)
    pairs = pd.DataFrame({"h": h[fin], "y": y[fin]}).drop_duplicates()
    conflict = np.zeros(len(df), dtype=bool)
    conflict[fin] = np.isin(h[fin], pairs["h"][pairs["h"].duplicated()].to_numpy())

    alive = np.flatnonzero(finite & ~conflict)
    dup = np.zeros(len(df), dtype=bool)
    dup[alive] = pd.Series(h[alive]).duplicated().to_numpy()
    keep = finite & ~conflict & ~dup
    del h, pairs, fin, alive  # free ~1 GB of temporaries before the frame copy

    out = df.loc[keep, [c for c in df.columns if c not in leak]].reset_index(drop=True)
    if groups is not None:
        out["group"] = groups.to_numpy()[keep]

    def count(mask):
        return pd.Series(y[mask]).value_counts()
    per = pd.DataFrame({"rows_in": pd.Series(y).value_counts(), "nonfinite": count(~finite),
                        "conflict_rows": count(conflict), "duplicates": count(dup),
                        "rows_out": count(keep)}).fillna(0).astype("int64")
    report = {
        "dataset": dataset,
        "rows_in": len(y),
        "nonfinite": int((~finite).sum()),
        "conflict_rows": int(conflict.sum()),
        "duplicates": int(dup.sum()),
        "rows_out": int(keep.sum()),
        "per_class": per,
        "kept_rows": np.flatnonzero(keep),  # positions in the input frame, row i of `out` = input kept_rows[i]
    }
    return out, report


def weight_follows_number(df):
    """CICIoT `Weight` leaks like `Number` if it is a deterministic function of it."""
    return not df[["Number", "Weight"]].drop_duplicates()["Number"].duplicated().any()
