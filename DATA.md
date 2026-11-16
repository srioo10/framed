# Datasets

All four are public. Place them where `src/nids/load.py` looks (`data/raw/`, or set `NIDS_DATA_DIR`), then the
loaders rebuild our cleaning and splits (cached under `data/processed/`).

| Paper name | Release used | Loader | Expected files | Source |
|---|---|---|---|---|
| CIC (CSE-CIC-IDS2018) | NetFlow v3 feature set, `NF-CICIDS2018-v3.csv` | `load.py` `NF_FILES["nf_cic"]` | `data/raw/netflow_v3/<id>/data/NF-CICIDS2018-v3.csv` | University of Queensland NetFlow datasets (Sarhan et al.) |
| ToN-IoT | NetFlow v3, `NF-ToN-IoT-v3.csv` | `NF_FILES["nf_ton"]` | `data/raw/netflow_v3/<id>/data/NF-ToN-IoT-v3.csv` | same |
| CTU-13 | bidirectional `.binetflow`, scenarios 1-13 | `load.py` `ctu13()` | `data/raw/ctu13/CTU-13-Dataset/<n>/*.binetflow` | Stratosphere Lab, CTU Prague |
| UGR'16 | weekly flows, weeks `june_week3` ... `august_week2` | `load.py` `ugr16()` | `data/raw/ugr16/<week>.parquet` | NESG, University of Granada (**parquet conversion: source to be stated by the authors**) |

SHA-256 of the files we used:

```
a2feb2d6253c2560fbffdfcc3e92b6c9235f1ea7c43a5f8d7902d7e352e84171  CTU-13-Dataset/4/capture20110815.binetflow
ec67980705e59d794eb2cea5a39d6ea23507a531036e1a8a6cc16c8985d15da9  CTU-13-Dataset/5/capture20110815-2.binetflow
8ae44d50e48fdff5b810b741b7824999ccf474081676562a54aa2b23d55d556f  CTU-13-Dataset/10/capture20110818.binetflow
b1b49932c7ce0c4e344431fbc229e4f91be06772abfe7a2e6e7d4b0d439d61d7  CTU-13-Dataset/6/capture20110816.binetflow
7974f3cb3f4b707d7543194b6cd96c52af804d4a11aad2645036fd6f3e635abe  CTU-13-Dataset/9/capture20110817.binetflow
a5e7311e25ff280e302ee5786b1aabf48dacfa6edaf25412f3aa1adbe1006039  CTU-13-Dataset/3/capture20110812.binetflow
366b87099b5cbbf41b7464eefccc224b10497101a4597f4f5dab6b8259aa3cf8  CTU-13-Dataset/2/capture20110811.binetflow
a4dbb317f53eaaef2538d1cc1cd228bda3743a84779032601d5762742c4daad1  CTU-13-Dataset/13/capture20110815-3.binetflow
d5fafadc606100b3dc4080dbf851a40d78c522c222576a0bcc375cbf13e0c106  CTU-13-Dataset/1/capture20110810.binetflow
615e28233cb3c494b74c1016af940394c2765df9a285413256e65b0ebd11bbca  CTU-13-Dataset/12/capture20110819.binetflow
f224baab443b56b2f0812cf2f9cdbb0f118b8af70095c196d9ead6a92c72ef0f  CTU-13-Dataset/7/capture20110816-2.binetflow
c2ff96c8908f1b24b2ff9dc2033ca9fd034b994f3a59e0cc4f1f53a0615c4ec0  CTU-13-Dataset/8/capture20110816-3.binetflow
d761b4513ee42be57193fe8f442e9234e73ac81f4fbd01248e37c23ca8ec1cf2  CTU-13-Dataset/11/capture20110818-2.binetflow
4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5  NF-UNSW-NB15-v3.csv
53ec8f468a43ede9b1536fabc0390af2fa33ab4312b23ce4d864f186a4651f78  NF-ToN-IoT-v3.csv
242a6971cc801eae621b1fc4d966db2cd0af9cc866805f36a6fc5d0058dfbb74  NF-CICIDS2018-v3.csv
```
