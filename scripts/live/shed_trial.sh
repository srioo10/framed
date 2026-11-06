#!/bin/bash
# One SHED trial (PREREG SHED live amendment): reset gateway, count SYNs reaching the Pi, flood from the laptop,
# save the gateway log window. Usage: shed_trial.sh <arm> <rate> <rep>   (run on the laptop, joined to bigp-iot)
set -e
ARM=$1 RATE=$2 REP=$3 DUR=120
OUT=runs/live/shed; mkdir -p $OUT
TAG=${ARM}_${RATE}_${REP}
PI=pi4r8@10.42.0.88
# keep wlo1 on bigp-iot (it roams back to IIITDM otherwise); internet stays on the phone tether
iw dev wlo1 link 2>/dev/null | grep -q "SSID: bigp-iot" || nmcli con up bigp-iot-attack ifname wlo1 >/dev/null 2>&1
ip route get 192.168.77.1 2>/dev/null | grep -q "via 192.168.4.1" || { echo "NO LAB ROUTE"; exit 1; }
# stop logger FIRST (it holds the port), then find the gateway by MAC, reset to run mode, point logger, restart
ssh -o BatchMode=yes $PI 'sudo -n systemctl stop onepass-gwlog.service; sleep 1; G="";
  for p in 1.1 1.2; do P=/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:$p:1.0-port0;
    /opt/onepass/live/venv/bin/esptool --chip esp32 -p $P read_mac 2>/dev/null | grep -q 94:54:c5:6f:b0:b4 && G=$P; done;
  [ -n "$G" ] || { echo NO_GATEWAY; exit 1; };
  /opt/onepass/live/venv/bin/esptool --chip esp32 -p $G --after hard-reset flash-id >/dev/null 2>&1;
  sudo -n sed -i "s#--port [^ ]* --out#--port $G --out#" /etc/systemd/system/onepass-gwlog.service; sudo -n systemctl daemon-reload;
  sudo -n systemctl start onepass-gwlog.service'
sleep 25  # gateway boot + sensor rejoin
T0=$(ssh -o BatchMode=yes $PI 'date +%s%3N')
ssh -o BatchMode=yes $PI "sudo -n timeout $((DUR + 10)) tcpdump -i wlan0 -nn -q 'tcp[tcpflags] & tcp-syn != 0 and tcp[tcpflags] & tcp-ack == 0 and dst host 192.168.77.1 and dst portrange 2000-3000' 2>/dev/null | wc -l" > $OUT/$TAG.syn &
sleep 2
.venv/bin/python scripts/live/flood.py 192.168.77.1 $RATE $DUR --ports 2000-3000 > $OUT/$TAG.sent
wait
sleep 40  # let idle flows close and the gateway log its decisions
T1=$(ssh -o BatchMode=yes $PI 'date +%s%3N')
ssh -o BatchMode=yes $PI "awk -F, -v a=$T0 -v b=$T1 '\$1>=a && \$1<=b' /opt/onepass/live/gw/gw-\$(ls /opt/onepass/live/gw | grep -oE '[0-9]{8}' | sort | tail -1).log" > $OUT/$TAG.gw
echo "$TAG sent: $(cat $OUT/$TAG.sent) | syn at Pi: $(cat $OUT/$TAG.syn) | gw lines: $(wc -l < $OUT/$TAG.gw)"
