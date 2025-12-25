#!/bin/bash

# Enable Wake-on-LAN (Magic Packet) for the WiFi connection 'WuTang_Lan'
# Run this script ON THE SERVER (this machine).

echo "Configuring Wake-on-LAN for 'WuTang_Lan'..."

# Modify the connection to enable WoWLAN with magic packet
# We use sudo because modifying system connections requires it
sudo nmcli connection modify "WuTang_Lan" 802-11-wireless.wake-on-wlan magic

if [ $? -eq 0 ]; then
    echo "Successfully enabled Wake-on-LAN on NetworkManager connection."
else
    echo "Error modifying connection. Ensure you have sudo rights."
    exit 1
fi

# Bring the connection up again to ensure settings apply (though modify often applies immediately)
# sudo nmcli connection up "WuTang_Lan" 

# Verify the setting
echo "Verifying configuration..."
STATUS=$(nmcli -f 802-11-wireless.wake-on-wlan connection show "WuTang_Lan")
echo "Current Setting (should say 'magic'): $STATUS"

echo ""
echo "Note: Ensure 'Wake on Wireless LAN' is enabled in your BIOS/UEFI."
