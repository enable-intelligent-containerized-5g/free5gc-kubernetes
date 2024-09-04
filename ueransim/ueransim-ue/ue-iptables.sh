#!/bin/bash

# Interfaces and routes
declare -A INTERFACE_ROUTE_MAP=(
    ["uesimtun0"]="10.1.0.0/16 192.168.1.0/24"  # Multiple interfaces
    ["uesimtun1"]="10.2.0.0/16"                 # One interface
    # Add more interfaces and routes
)

# Add an interface if not exist
add_route() {
    local interface=$1
    local route=$2
    
    # Check if the route exists
    ip route | grep -q "$route"
    if [ $? -ne 0 ]; then
        echo "Adding route $route dev $interface"
        ip route add "$route" dev "$interface"
    else
        echo "The route $route already exists"
    fi
}

# Infinite loop that repeats every 5 seconds
while true; do
    for interface in "${!INTERFACE_ROUTE_MAP[@]}"; do
        # List of routes by interface
        routes="${INTERFACE_ROUTE_MAP[$interface]}"

        # Check if the interface exists
        ip link show "$interface" > /dev/null 2>&1
        if [ $? -eq 0 ]; then
            echo "La interfaz $interface está activa"
            
            # Iterate over the routes to add each one
            for route in $routes; do
                add_route "$interface" "$route"
            done
        else
            echo "The interface $interface not exist"
        fi
    done

    # Wait 5 seconds
    sleep 5
done