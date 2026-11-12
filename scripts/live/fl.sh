#!/bin/bash
# flash a staged build: fl.sh gw|sensor <dir under ~/fw>
set -e
case "$1" in gw) P=/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.1:1.0-port0;; sensor) P=/dev/serial/by-path/platform-fd500000.pcie-pci-0000:01:00.0-usb-0:1.2:1.0-port0;; *) exit 2;; esac
D=~/fw/$2; APP=$(ls $D/*.bin | grep -v -e bootloader -e partition)
/opt/onepass/live/venv/bin/esptool.py --chip esp32 -p $P -b 460800 write_flash 0x1000 $D/bootloader.bin 0x8000 $D/partition-table.bin 0x10000 $APP 2>&1 | grep -E "Hash of data verified|rror" | tail -2
