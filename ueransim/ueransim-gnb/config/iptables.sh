#!/bin/bash

### Implement networking rules
ip route add 20.20.20.1/32 dev lb via 10.10.2.1 && sleep infinity