#!/bin/bash 

# Implement networking rules (Only with loxilb)
# ip route add 20.20.20.1/32 dev n2 via 10.10.2.1
# sleep infinity

# run gnb
/ueransim/build/nr-gnb --config /ueransim/config/free5gc-gnb.yaml

    
