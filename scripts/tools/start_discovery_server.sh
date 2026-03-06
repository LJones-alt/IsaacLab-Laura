#!/bin/bash

# This script starts a ROS 2 Discovery Server.
# Use this when communicating over a VPN where multicast is blocked.

# Get the IP address of the machine on the VPN (e.g., tailscale0, tun0)
# You can also manually set this if the auto-detection fails.
VPN_INTERFACE=$(ip addr | grep -e "tailscale" -e "tun" | awk '{print $NF}' | head -n 1)

if [ -z "$VPN_INTERFACE" ]; then
    echo "[WARNING] No VPN interface (tailscale, tun) detected. Falling back to eth0."
    VPN_INTERFACE="eth0"
fi

IP_ADDR=$(ip addr show $VPN_INTERFACE | grep "inet " | awk '{print $2}' | cut -d/ -f1)

if [ -z "$IP_ADDR" ]; then
    echo "[ERROR] Could not detect IP address for interface $VPN_INTERFACE."
    exit 1
fi

PORT=11811

echo "[INFO] Starting ROS 2 Discovery Server on $IP_ADDR:$PORT (Interface: $VPN_INTERFACE)"
echo "[INFO] To connect other machines, run: export ROS_DISCOVERY_SERVER=$IP_ADDR:$PORT"

# Source ROS 2 if not already sourced
if [ -f "/opt/ros/humble/setup.bash" ]; then
    source /opt/ros/humble/setup.bash
fi

fastdds discovery --server-id 0 --ip-address $IP_ADDR --port $PORT
