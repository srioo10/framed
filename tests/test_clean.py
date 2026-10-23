import numpy as np
import pandas as pd

from nids import clean


def _frame():
    # rows 0,1: exact duplicates (same label)  -> keep one
    # rows 2,3: same features, different label -> both dropped (conflict)
    # row 4: NaN, row 5: inf                    -> dropped
    # row 6: unique
    return pd.DataFrame({
        "a": [1.0, 1.0, 2.0, 2.0, np.nan, 3.0, 4.0],
        "b": [0.0, 0.0, 5.0, 5.0, 1.0, np.inf, 6.0],
        "Number": [9, 9, 9, 9, 9, 9, 9],
        "y": ["x", "x", "x", "z", "x", "x", "z"],
        "family": ["F", "F", "F", "G", "F", "F", "G"],
        "src": ["s1"] * 7,
    })


def test_clean_drops_nonfinite_conflicts_and_duplicates():
    out, rep = clean.clean(_frame(), "ciciot_paper")
    assert sorted(out[["a", "b"]].itertuples(index=False, name=None)) == [(1.0, 0.0), (4.0, 6.0)]
    assert rep["rows_in"] == 7
    assert rep["nonfinite"] == 2
    assert rep["conflict_rows"] == 2
    assert rep["duplicates"] == 1
    assert rep["rows_out"] == 2


def test_clean_drops_leak_columns_before_hashing():
    # rows differ only in the leak column -> duplicates once it is dropped
    df = pd.DataFrame({"a": [1.0, 1.0], "Number": [1, 2], "y": ["x", "x"], "family": ["F", "F"], "src": ["s", "s"]})
    out, rep = clean.clean(df, "ciciot_paper")
    assert "Number" not in out.columns
    assert rep["duplicates"] == 1 and len(out) == 1


def test_clean_keeps_labels_and_adds_group_only_where_defined():
    out, _ = clean.clean(_frame(), "ciciot_csv")
    assert {"y", "family", "src", "group"} <= set(out.columns)
    out, _ = clean.clean(_frame(), "ciciot_paper")
    assert "group" not in out.columns


def test_clean_per_class_report_sums_to_totals():
    _, rep = clean.clean(_frame(), "ciciot_paper")
    per = rep["per_class"]
    assert per["rows_in"].sum() == rep["rows_in"]
    assert per["rows_out"].sum() == rep["rows_out"]


def test_group_ids_follow_preregistration():
    bot = pd.DataFrame({"pkSeqID": [1, 9_999, 10_000, 25_000]})
    assert list(clean.group_ids(bot, "bot_iot")) == [0, 0, 1, 2]
    hour = 394_979 * 3600
    unsw = pd.DataFrame({"stime": [hour, hour + 3599, hour + 3600]})
    g = clean.group_ids(unsw, "unsw_raw")
    assert g.iloc[0] == g.iloc[1] != g.iloc[2]


def test_weight_follows_number():
    df = pd.DataFrame({"Number": [1, 1, 2, 2], "Weight": [5.0, 5.0, 7.0, 7.0]})
    assert clean.weight_follows_number(df)
    df.loc[3, "Weight"] = 8.0
    assert not clean.weight_follows_number(df)
