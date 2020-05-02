#!/bin/bash
export LD_PRELOAD=/usr/lib/arm-linux-gnueabihf/libatomic.so.1
sudo dhclient wlan0
python3 /home/pi/Desktop/cigar_counter/main.py
ret=$?
echo $ret
bash
if [ $ret -ne 0 ]; then
    sleep 60
    reboot
fi
