#!/bin/bash

mkdir /dev/net
mknod /dev/net/tun c 10 200

nohup /ueransim/config/ue-iptables.sh &
nohup /ueransim/nr-ue -c /ueransim/config/ue3.yaml 
