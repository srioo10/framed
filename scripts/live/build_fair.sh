#!/bin/bash
# PREREG LIVE-FAIR: build the two gateway firmwares with CONFIG_GW_TAU = tau_fair (nothing else changed) and stage them
# on Pi #1 as ~/fw/fair_veto and ~/fw/fair_ord. Usage: build_fair.sh <tau> [--no-stage]
set -e
T=$1; [[ $T =~ ^0\.[0-9]+$ ]] || { echo "tau must look like 0.xxx"; exit 2; }
cd "$(dirname "$0")/../../firmware/gateway"
. ~/esp/esp-idf/export.sh >/dev/null 2>&1
for b in veto ord; do
  src=sdkconfig.sk_nat; [ $b = ord ] && src=sdkconfig.sk_nat_ord
  grep -q '^CONFIG_GW_TAU=' $src || { echo "no GW_TAU in $src"; exit 3; }
  sed "s/^CONFIG_GW_TAU=.*/CONFIG_GW_TAU=\"$T\"/" $src > sdkconfig.fair_$b   # sdkconfig* hold lab creds: gitignored
  idf.py -B build_fair_$b -D SDKCONFIG=sdkconfig.fair_$b build >/tmp/claude-1000/build_fair_$b.log 2>&1 || { echo "build $b failed"; exit 4; }
  diff <(grep -v '^#' $src | grep -v GW_TAU) <(grep -v '^#' sdkconfig.fair_$b | grep -v GW_TAU) >/dev/null || { echo "sdkconfig drift in $b"; exit 5; }
  echo "built $b tau=$(grep '^CONFIG_GW_TAU' sdkconfig.fair_$b | cut -d= -f2)"
  if [ "$2" != --no-stage ]; then
    D=build_fair_$b; ssh pi4r8@10.42.0.88 "mkdir -p ~/fw/fair_$b"
    scp -q $D/bootloader/bootloader.bin $D/partition_table/partition-table.bin $D/veto_gateway.bin pi4r8@10.42.0.88:fw/fair_$b/
    echo "staged fair_$b"
  fi
done
