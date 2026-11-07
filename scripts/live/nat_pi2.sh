#!/usr/bin/env bash
# PREREG LIVE-NAT: make Pi #2 a NAT router so the victim (on Pi #2) and the wired attacker laptop share one address
# on the gateway's IoT Wi-Fi. Join the IoT Wi-Fi first with your own nmcli command (no credentials in this file).
# Usage (on Pi #2, as root): bash nat_pi2.sh up   [wlan=wlan0] [lan=eth0]
#                           bash nat_pi2.sh down [wlan=wlan0] [lan=eth0]
set -eu
CMD="${1:-}"; WLAN="${2:-wlan0}"; LAN="${3:-eth0}"; LANNET=10.99.0.0/24; LANIP=10.99.0.1/24
case "$CMD" in
  up)
    ip -4 addr show "$WLAN" | grep -q 'inet 192\.168\.4\.' || { echo "$WLAN has no 192.168.4.x lease: join the IoT Wi-Fi first"; exit 1; }
    ip addr add "$LANIP" dev "$LAN" 2>/dev/null || true
    ip link set "$LAN" up
    sysctl -qw net.ipv4.ip_forward=1
    iptables -t nat -C POSTROUTING -s "$LANNET" -o "$WLAN" -j MASQUERADE 2>/dev/null || \
      iptables -t nat -A POSTROUTING -s "$LANNET" -o "$WLAN" -j MASQUERADE
    iptables -C FORWARD -i "$LAN" -o "$WLAN" -j ACCEPT 2>/dev/null || iptables -A FORWARD -i "$LAN" -o "$WLAN" -j ACCEPT
    iptables -C FORWARD -i "$WLAN" -o "$LAN" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || \
      iptables -A FORWARD -i "$WLAN" -o "$LAN" -m state --state RELATED,ESTABLISHED -j ACCEPT
    echo "NAT up: $LAN ($LANIP) -> $WLAN $(ip -4 -o addr show "$WLAN" | awk '{print $4}')"
    echo "On the laptop: give its wired port 10.99.0.2/24 and route the lab via Pi #2, e.g."
    echo "  sudo ip addr add 10.99.0.2/24 dev <eth>; sudo ip route add 192.168.77.0/24 via 10.99.0.1"
    echo "  for k in \$(seq 100 199); do sudo ip route add 192.168.\$k.0/24 via 10.99.0.1; done"
    ;;
  down)
    iptables -t nat -D POSTROUTING -s "$LANNET" -o "$WLAN" -j MASQUERADE 2>/dev/null || true
    iptables -D FORWARD -i "$LAN" -o "$WLAN" -j ACCEPT 2>/dev/null || true
    iptables -D FORWARD -i "$WLAN" -o "$LAN" -m state --state RELATED,ESTABLISHED -j ACCEPT 2>/dev/null || true
    ip addr del "$LANIP" dev "$LAN" 2>/dev/null || true
    echo "NAT down"
    ;;
  *) echo "usage: $0 up|down [wlan] [lan]"; exit 1;;
esac
