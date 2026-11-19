# Frame-proof blocking: artifact

Three things a reader can run without our hardware.

| Command | What it checks | Time |
|---|---|---|
| `bash artifact/check.sh` | Replays every logged live gateway decision through the deployed C decision path (0 mismatches expected) and re-runs the framing-witness search on the live logs. | ~1 min |
| `python scripts/frame_audit.py flows.csv --detector frame_audit:rule_f2b` | Framing audit of any detector on your own benign NetFlow: share of innocent host-days an in-budget injector (m flows per key per 240 s) can make it alarm on. Plug in your detector as `module:function`. | minutes |
| `python scripts/live/witness.py <log dir> <build> <tau> --v2` | For each alarm the certificate could not back, the at most m context-writing connections whose removal clears it (a replay-checkable counterfactual). | ~10 s per build |

Live logs: `results/live/*/gw-*.log` (one `F` line per closed flow: time, 5-tuple, bytes, packets, score, lower bound,
alarm, block), run windows in `fair_runs.txt`, attacker actions in `fair_inject.log`. Lab addresses only.
The ESP32 gateway firmware is in `firmware/gateway`; `engine/certb.h` is the portable certificate core (no malloc).
Pre-registration log of every test, including the failed ones: `PREREG.md`.
