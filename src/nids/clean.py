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
    """Split groups: capture file, 1-hour stime window, or 10k-pkSeqID block. None = no group id."""
    if dataset == "ciciot_csv":
        return df["src"]
    if dataset == "unsw_raw":
        return df["stime"] // 3600
    if dataset == "bot_iot":
        return df["pkSeqID"] // 10_000
    return None


def feature_columns(df):
    return [c for c in df.columns if c not in LABELS]


def row_hash(df):
    return pd.util.hash_pandas_object(df[feature_columns(df)], index=False).to_numpy()


def clean(df, dataset):
    """Return (cleaned frame, report). Report counts are rows removed at each step, overall and per class."""
    y_in = df["y"].astype(str).to_numpy()
    groups = group_ids(df, dataset)
    df = df.drop(columns=[c for c in LEAK_COLS[dataset] if c in df.columns])
    if groups is not None:
        df["group"] = groups.to_numpy()

    feats = df[feature_columns(df)]
    num = feats.select_dtypes("number").to_numpy(dtype="float64", na_value=np.nan)
    finite = np.isfinite(num).all(axis=1) & feats.select_dtypes(exclude="number").notna().all(axis=1).to_numpy()
    df, y = df[finite], y_in[finite]

    h = row_hash(df)
    pairs = pd.DataFrame({"h": h, "y": y}).drop_duplicates()
    conflict = np.isin(h, pairs["h"][pairs["h"].duplicated()].to_numpy())
    df, h, y = df[~conflict], h[~conflict], y[~conflict]

    dup = pd.Series(h).duplicated().to_numpy()
    out = df[~dup].reset_index(drop=True)

    y_fin = y_in[finite]
    per = pd.DataFrame({
        "rows_in": pd.Series(y_in).value_counts(),
        "nonfinite": pd.Series(y_in[~finite]).value_counts(),
        "conflict_rows": pd.Series(y_fin[conflict]).value_counts(),
        "duplicates": pd.Series(y[dup]).value_counts(),
        "rows_out": pd.Series(y[~dup]).value_counts(),
    }).fillna(0).astype("int64")

    report = {
        "dataset": dataset,
        "rows_in": len(y_in),
        "nonfinite": int((~finite).sum()),
        "conflict_rows": int(conflict.sum()),
        "duplicates": int(dup.sum()),
        "rows_out": len(out),
        "per_class": per,
    }
    return out, report


def weight_follows_number(df):
    """CICIoT `Weight` leaks like `Number` if it is a deterministic function of it."""
    return bool(df.groupby("Number", observed=True)["Weight"].nunique().max() == 1)
