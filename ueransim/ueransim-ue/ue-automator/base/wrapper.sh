#!/bin/bash

mkdir /dev/net
mknod /dev/net/tun c 10 200
# chmod 755 /dev/net/tun

# nohup /ueransim/config/ue-iptables.sh &
# nohup /ueransim/nr-ue -c /ueransim/config/ue.yaml

/ueransim/nr-ue -c /ueransim/config/ue.yaml