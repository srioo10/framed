# Scripts

All scripts stay in this folder because the paper's scripts import some of the older ones. The tables below say
which phase each script belongs to. Run from the repository root with `PYTHONPATH=src`.

## Paper: host-context poisoning and Veto

| Script | What it does |
|---|---|
| `gate_attack.py` | Main harness: trains Veto and RF / XGBoost, runs the one-way, shared-key (`--established`), adaptive, over-budget, throttle, optimiser (`--opt-harm`) and padding (`--dilute`) attacks; reports harm, certified detection and violations |
| `spoof_attack.py` | Injection generator (K1 / K14): spoofed flows within budget m, victim selection, published baselines |
| `kitsune_native.py` | Attack on unmodified Kitsune-py (AfterImage + KitNET) on its Mirai capture; `r` suffix = shared-key path |
| `harm_ci.py` | 95% intervals for harm and the zero-violation results |
| `bowl.py`, `fighters.py`, `arena.py`, `tabm.py` | Model search arena, deletion box (`dbox`), IBP lower bound, certified training, TabM |
| `boss_bowl.py`, `ultra_bosses.py`, `gate_bosses.py` | Tuned and size-matched tree baselines (K9), gated trees |
| `cert_trees.py`, `k5_trees.py`, `k5b_net.py` | Tree box bounds, exact MILP verifier and robust-trained trees (K5); branch and bound for Veto (K5b) |
| `conf_pairs.py`, `conf_clean.py`, `conf_baselines.py`, `conf_analyze.py` | Confirmatory protocol (CONF): splits, clean and certified results, published baselines, analysis |
| `ips_point.py`, `k1_unsw.py`, `k7_longmem.py`, `k8_ops.py`, `k10_capacity.py`, `check_cost.py` | Kill tests K15, F4 (UNSW-NB15), K7, K8, K10, checking cost |
| `audit_f1.py`, `audit_f2f3.py` | Audit fixes: float64 bound with outward margin, bootstrap intervals |
| `engine_export_b.py`, `engine_k11.py`, `engine_flood.py` | Export Veto to the C engine; state-exhaustion (K11) and flood tests |
| `probe_cert.py`, `probe_rdet.py`, `probe_robust.py`, `probe_trees_gate.py` | Exploratory probes that led to certifying the block (disclosed in `PREREG.md`) |
| `kaggle_bowl.py`, `kaggle_conf.py`, `kaggle_k5.py`, `rebuild_round.py` | Kaggle launchers and recovery |
| `live/` | ESP32 live tests: flow meter, gateway log, injection |

## Shared helpers (used by both phases)

`score_bank.py`, `host_stream.py`, `sensitivity.py`, `roc.py`, `ple.py`, `bank_report.py`, `dataset_audit.py`,
`leak_controls.py`.

## Earlier phase: ARLHNIDS re-evaluation and OnePass-IDS

Kept for the record; their saved outputs are in `archive/results/` and their tests in `PREREG.md`.

`arlhnids_attrib.py`, `arlhnids_sids.py`, `audit_clean.py`, `backoff_ciciot.py`, `baseline_sanity.py`, `calib.py`,
`candidates.py`, `crn_gate_qcad.py`, `crn_gate_refclass.py`, `crn_gates.py`, `discover.py`, `engine_export.py`,
`feature_audit.py`, `fingerprint_ablation.py`, `flood_check.py`, `flood_detectors.py`, `flowtok.py`,
`flowtok_eval.py`, `forward.py`, `fusion_diagnostics.py`, `host_audit.py`, `host_claim.py`, `host_report.py`,
`host_robust.py`, `hostctx_ref.py`, `leaf_diagnostic.py`, `model_eval.py`, `onepass_claims.py`, `onepass_cost.py`,
`paper_figs.py`, `pilot_lofo.py`, `stage5.py`, `stage5_fpr.py`, `test_lean_hostctx.py`, `tune_mlp.py`,
`adversarial.py`.
