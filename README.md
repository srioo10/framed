# framed

**Framing innocent devices by poisoning IDS host context, and Veto, a detector whose blocks are certified:
injected flows can never cause a block.**

Network intrusion detectors that also block (IPS mode) rely on host context: how many connections a host made in the
last few seconds or minutes, and what share of them failed or reached new ports and peers. These features are what
let a detector notice attacks it was never trained on. They also have a flaw. Anyone who can add flows to a host's
context can make an innocent host look like a scanner, and an IPS will then cut it off.

The attacker never touches the victim's own traffic. Two paths work:

- **Shared key.** The attacker sits behind the same NAT or CGNAT address or subnet as the victim. Its own truthful,
  fully established connections poison the context the victim is scored against. No spoofing is needed, and
  handshake or source-validation filters do not help.
- **Spoofed one-way flows,** where source validation is missing.

Veto is a small detector (a TabM ensemble of 8 members, 3.6 KB as int8) that blocks a flow only when it can prove the
block is not the attacker's doing. For each flow it computes the set of all clean contexts consistent with what it
observes, given at most m injected flows per host key per 240 s (the deletion box, Theorem 1). It then bounds its
score over that whole box with interval bound propagation (Theorem 2). It blocks only if the lower bound clears the
threshold. Otherwise it raises an alarm without blocking. Injection within the budget therefore cannot cause a
block (Corollary).

Every test, including the ones that failed, was written down before it ran, with its pass rule, in
[`PREREG.md`](PREREG.md).

## Main results

All of these are from pre-registered tests, with m = 10 flows per key per 240 s.

| Question | Result |
|---|---|
| Is the threat real? | Spoofed context makes published detectors (Kitsune-style, HorusEye-style, Deep SAD, iForest) and plain RF / XGBoost block 88–100% of innocent host-days on CTU-13 and UGR'16 |
| Without spoofing? | Shared-key, established-only injection: RF / XGBoost 9.4 / 13.8% of host-days on CTU-13, 92 / 98% on UGR'16 |
| On the original Kitsune code? | Unmodified Kitsune-py on its own Mirai capture: 13 of 13 victims alarmed (2 of 13 clean), both spoofed and shared-key. Single-packet flows were not enough (5 of 13, reported as a failure) |
| Does the certificate hold? | 0 injection-caused blocks over 72 attack patterns, an adaptive attacker and a white-box per-flow optimiser, on all datasets |
| What does it cost? | Veto's clean block rate is 0.55 / 2.0 / 1.33% of host-days (CTU-13 / UGR'16 / CIC), with 95% intervals containing the 1% budget. ToN-IoT drifts to 4.0% |
| Versus certified trees? | Veto leads robust-trained trees with an exact MILP verifier by +0.104 [0.051, 0.166] under the proof rule. Under the "undecided = blocked" convention the lead is +0.030, not significant |
| On a $5 gateway? | Inline on an ESP32 access point: 29 / 59 ms per flow, 117 KB heap. In monitor mode a within-budget spoof lifts the sensor over the threshold (an ordinary IPS would block it) while Veto's certified bound keeps it online; over budget Veto blocks too. In enforce the /24 shared-key attack is blunted within budget (it bites only over budget) and the spoof path is not stageable on our lab AP — matching the offline /24 result |

What it does not do (all reported in the paper and in `PREREG.md`):

- an attacker far over the budget can still cause blocks (20–66% of host-days at 5–10× the budget);
- a low-rate attacker looks like a spoofer, and no context certificate can block it on context alone (Proposition 1);
- padding the attacker's own context hides it on CTU-13 (a flow-only second channel fixes this, at a cost on CIC);
- the attack failed on UNSW-NB15;
- trees are not costlier to certify;
- Veto does not beat trees on clean detection;
- on real hardware the /24 (NAT) shared-key attack is weaker within budget than per-IP keying, and the spoof path is not stageable in enforce on a standard access point (source-IP anti-spoofing), so the live demonstration is sharpest with per-device keys.

## Reproducing the key results

Python 3.12, `pip install -r requirements.txt`. Datasets are NetFlow-v3 (CTU-13, UGR'16, CSE-CIC-IDS2018, ToN-IoT);
loaders and leak-free cleaning are in `src/nids/`.

| Result | Command | Output |
|---|---|---|
| Attack on all detectors, certificate soundness, harm | `PYTHONPATH=src python scripts/gate_attack.py nf_ctu13 --tau-day --genome '<B>'` | `results/cert/gate_k4_*_day.csv` |
| Shared-key attacker | same, with `--established` | `..._sharedkey.csv` |
| White-box optimiser | same, with `--opt-harm` | `opt_harm_*.csv` |
| Padding | same, with `--dilute 0 10 30 100 300 1000` | `dilute_*.csv` |
| Native Kitsune | `python -I scripts/kitsune_native.py <tsv> 0x1 30x3` (add `r` for shared-key, `30x3r`) | `kitsune_native_*.json` |
| Certified trees (exact MILP) | `python scripts/k5_trees.py` | `k5_*.csv` |
| Harm intervals | `python scripts/harm_ci.py` | `harm_ci.json` |

Native Kitsune needs `git clone https://github.com/ymirsky/Kitsune-py third_party/Kitsune-py` (we used commit
`7c8ea51`, unmodified). `<B>` is the Veto genome, pinned in `PREREG.md` (CONF rows). The C engine (`engine/certb.h`, `engine/certids.c`) and
the ESP32 firmware (`firmware/gateway`, `firmware/sensor`) implement the same box and bound on the device.
`scripts/README.md` says which scripts belong to the paper and which to the earlier phase.

## How this started

The project began as a leak-free re-evaluation of ARLHNIDS-IoT and a small one-pass IDS (OnePass-IDS). While testing
its C engine we found that spoofed flows made neural and KitNET detectors alarm on benign hosts through their host
context. That observation became this work. The earlier study, its results and its failed directions are kept in
[`archive/`](archive/) and in the dated rows of `PREREG.md`.
