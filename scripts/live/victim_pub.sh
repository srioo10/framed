#!/usr/bin/env bash
# PREREG LIVE-NAT victim: an IoT-sensor-like publisher on Pi #2 (behind the NAT it shares with the attacker).
# Publishes one MQTT message to the lab broker every PERIOD seconds and logs ok/fail per publish, so a block on the
# shared address shows up as failed publishes. Lab network only.
# Usage (on Pi #2): bash victim_pub.sh [broker=192.168.77.1] [period_s=15] [log=victim_pub.log]
BROKER="${1:-192.168.77.1}"; PERIOD="${2:-15}"; LOG="${3:-victim_pub.log}"
case "$BROKER" in 192.168.77.*) ;; *) echo "lab broker only (192.168.77.x)"; exit 1;; esac
if command -v mosquitto_pub >/dev/null; then
  pub() { timeout 5 mosquitto_pub -h "$BROKER" -p 1883 -t lab/victim -m "$(date +%s)" -q 0; }
else  # fallback: one TCP connection to the broker port per period (still one metered flow per publish)
  pub() { timeout 5 bash -c "exec 3<>/dev/tcp/$BROKER/1883" 2>/dev/null; }
  echo "mosquitto_pub not found: using TCP-connect fallback" | tee -a "$LOG"
fi
echo "$(date -u +%FT%TZ) start broker=$BROKER period=$PERIOD" | tee -a "$LOG"
while true; do
  if pub; then r=ok; else r=FAIL; fi
  echo "$(date -u +%FT%TZ) $(date +%s%3N) $r" >> "$LOG"
  sleep "$PERIOD"
done
