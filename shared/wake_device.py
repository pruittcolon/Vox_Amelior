import socket
import struct


def wake_on_lan(macaddress):
    """Switches on remote computers using WOL."""

    # Check if macaddress has the correct format
    if len(macaddress) == 17:
        sep = macaddress[2]
        macaddress = macaddress.replace(sep, "")
    elif len(macaddress) != 12:
        raise ValueError("Incorrect MAC address format")

    # Pad the synchronization stream.
    data = b"FFFFFFFFFFFF" + (macaddress * 16).encode()
    send_data = b""

    # Split up the hex values and pack.
    for i in range(0, len(data), 2):
        send_data += struct.pack(b"B", int(data[i : i + 2], 16))

    # Broadcast it to the LAN.
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    sock.sendto(send_data, ("<broadcast>", 9))
    print(f"Magic Packet sent to {macaddress}")


if __name__ == "__main__":
    # MAC Address of the device (Nemo Server) on wlo1
    # 80:32:53:22:c1:35
    target_mac = "80:32:53:22:c1:35"

    print(f"Sending Magic Packet to wake device {target_mac}...")
    try:
        wake_on_lan(target_mac)
        print("Done.")
    except Exception as e:
        print(f"Error sending packet: {e}")
