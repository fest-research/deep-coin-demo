#!/bin/bash
docker run --net=host --device /dev/video0 -v /dev:/dev:rw -v /home/pirate:/notebooks/deep:rw -v /sys/class/gpio:/sys/class/gpio:rw  -v /sys/bus:/sys/bus:rw --privileged -d fest/deep-rpi sleep 360000
