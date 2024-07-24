#!/bin/bash 

    ### Implement networking rules
    ip route add 20.20.20.1/32 dev n2 via 10.10.2.1
    sleep infinity

    ### run gnb
    /ueransim/nr-gnb --config /ueransim/config/free5gc-gnb.yaml

    