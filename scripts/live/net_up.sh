#!/bin/bash
# LIVE-NAT-ENF lab plumbing on Pi #1 (idempotent). Attacker sockets bound to wlsta (bigp-nat, behind the sensor NAPT)
# must leave via wlsta even toward Pi-local peers, so: source rule before the local table, accept_local on wlsta.
A=$(ip -4 -o addr show wlsta | awk "{print \$4}" | cut -d/ -f1); [ -n "$A" ] || { echo "wlsta has no address"; exit 1; }
ip route replace default via 192.168.5.1 dev wlsta table 100
ip rule show | grep -q "from $A lookup 100" || ip rule add pref 50 from $A lookup 100
ip rule show | grep -q "^100:.*lookup local" || ip rule add pref 100 lookup local
ip rule show | grep -q "^0:.*lookup local" && ip rule del pref 0 lookup local
sysctl -qw net.ipv4.conf.wlsta.accept_local=1 net.ipv4.conf.wlsta.rp_filter=0 net.ipv4.conf.all.rp_filter=0 net.ipv4.conf.wlan0.rp_filter=0
# LIVE-FAIR F4 fix (15 Nov): in D runs the attacker (wlsta on the gateway AP) has a Pi-owned 192.168.4.x address, so
# its scan reaches wlan0 with a local source (accept it) and the Pi's replies must go back through the gateway, not
# short-cut to wlsta (rule for locally generated packets only, before the local table).
sysctl -qw net.ipv4.conf.wlan0.accept_local=1
ip route replace 192.168.4.0/24 via 192.168.77.2 dev wlan0 table 102
ip rule show | grep -q "lookup 102" || ip rule add pref 48 to 192.168.4.0/24 iif lo lookup 102
for k in 120 121 122 123; do ip addr show dev wlan0 | grep -q "192.168.$k.1/32" || ip addr add 192.168.$k.1/32 dev wlan0; done
echo "attacker $A"; ip rule show
