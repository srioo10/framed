"""Loaders must reproduce the row counts printed in ARLHNIDS Tables 2-4 exactly."""
import pytest

from nids import load


def test_unsw_official_matches_table2():
    df = load.unsw_official()
    assert (df.src == "train").sum() == 175_341
    assert (df.src == "test").sum() == 82_332
    counts = df.y.value_counts()
    assert counts["Normal"] == 93_000 and counts["Worms"] == 174 and counts["Generic"] == 58_871
    assert df.family.notna().all()


def test_bot_iot_matches_table3():
    df = load.bot_iot()
    assert (df.src == "train").sum() == 2_934_817
    assert (df.src == "test").sum() == 733_705
    counts = df.y.value_counts()
    assert counts["DDoS"] == 1_926_624 and counts["Normal"] == 477 and counts["Theft"] == 79
    assert df.family.notna().all()


def test_ciciot_paper_matches_table4():
    df = load.ciciot_paper()
    expected = {"DDoS": 137_254, "Normal": 129_832, "Spoofing": 59_854, "DoS": 40_686,
                "Mirai": 15_090, "Brute Force": 13_064, "Web-Based": 3_218, "Recon": 2_262}
    assert df.y.value_counts().to_dict() == expected
    assert len(df) == 401_260
    assert df.shape[1] == 39 + 3  # 39 features + y, family, src


@pytest.mark.parametrize("label,family", [
    ("DDoS-RSTFINFlood", "Flood"), ("DoS-HTTP_Flood", "Flood"), ("Mirai-greip_flood", "Mirai"),
    ("VulnerabilityScan", "Recon"), ("DNS_Spoofing", "Spoofing"), ("DictionaryBruteForce", "BruteForce"),
    ("BenignTraffic", "Benign"), ("Uploading_Attack", "Web"), ("XSS", "Web"),
])
def test_cic_family(label, family):
    assert load.cic_family(label) == family
