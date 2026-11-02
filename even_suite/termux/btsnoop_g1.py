#!/usr/bin/env python3
"""Show what an app sent to the Even G1 glasses, from Android's Bluetooth log.

Usage:
  python3 btsnoop_g1.py <bugreport.zip | btsnoop_hci.log> [--all] [--cmd 26]

Turn on Developer options > "Enable Bluetooth HCI snoop log" (Enabled, not
Filtered), restart Bluetooth, use the app, then Developer options > "Take bug
report". The report (a zip) holds FS/data/misc/bluetooth/logs/btsnoop_hci.log.

Prints every write to the glasses and every notification back, decoded where
the command is known. Heartbeats (0x25) are hidden unless --all. --cmd 26
shows only that command (hex).
"""
import struct
import sys
import zipfile
from datetime import datetime, timezone

ATT_CID = 0x0004
WRITE_CMD, WRITE_REQ, NOTIFY = 0x52, 0x12, 0x1B


def read_log(path):
    """The raw btsnoop bytes, from the log itself or the newest one in a bug report zip."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            names = [n for n in z.namelist() if "btsnoop_hci.log" in n and not n.endswith("/")]
            if not names:
                sys.exit("No btsnoop_hci.log in this bug report: was the snoop log Enabled (not Filtered)?")
            # The current log, not an older rotated one ("btsnoop_hci.log.last").
            names.sort(key=lambda n: (n.endswith(".last"), len(n)))
            return z.read(names[0])
    with open(path, "rb") as f:
        return f.read()


def records(data):
    """(time, sent_by_phone, h4 packet) for each record."""
    if data[:8] != b"btsnoop\0":
        sys.exit("Not a btsnoop log")
    pos = 16
    while pos + 24 <= len(data):
        orig, incl, flags, _drops, ts = struct.unpack(">IIIIq", data[pos:pos + 24])
        pos += 24
        pkt = data[pos:pos + incl]
        pos += incl
        # Microseconds since 0 AD.
        when = datetime.fromtimestamp((ts - 0x00E03AB44A676000) / 1e6, timezone.utc)
        yield when, (flags & 1) == 0, pkt


def att_packets(data):
    """Reassembled ATT PDUs: (time, sent, connection handle, pdu), plus the peer address of each handle."""
    partial = {}
    peers = {}
    for when, sent, pkt in records(data):
        if not pkt:
            continue
        kind, body = pkt[0], pkt[1:]
        if kind == 0x04 and len(body) > 3 and body[0] == 0x3E:  # LE meta event
            sub = body[2]
            if sub in (0x01, 0x0A) and body[3] == 0:  # (enhanced) connection complete, success
                handle = struct.unpack("<H", body[4:6])[0] & 0x0FFF
                addr = ":".join(f"{b:02X}" for b in reversed(body[8:14]))
                peers[handle] = addr
            continue
        if kind != 0x02 or len(body) < 4:
            continue
        hf, length = struct.unpack("<HH", body[:4])
        handle, pb = hf & 0x0FFF, (hf >> 12) & 0x3
        frag = body[4:4 + length]
        key = (handle, sent)
        if pb in (0, 2):  # start of an L2CAP frame
            if len(frag) < 4:
                continue
            l2len, cid = struct.unpack("<HH", frag[:4])
            partial[key] = [l2len, cid, bytearray(frag[4:]), when]
        elif pb == 1 and key in partial:
            partial[key][2] += frag
        else:
            continue
        l2len, cid, buf, start = partial[key]
        if len(buf) >= l2len:
            del partial[key]
            if cid == ATT_CID:
                yield start, sent, handle, bytes(buf[:l2len]), peers


def describe(v):
    """A G1 command in words, where it is known."""
    if not v:
        return ""
    c = v[0]
    if c == 0x26 and len(v) >= 8 and v[4] == 0x02:
        return f"DISPLAY POSITION state={v[5]} ({'preview/store' if v[5] else 'close'}) height={v[6]} depth={v[7]}"
    if c == 0x26:
        return f"0x26 hardware setting, sub {v[4] if len(v) > 4 else '?'}"
    if c == 0x4E and len(v) >= 9:
        text = v[9:].decode("utf-8", "replace").replace("\n", " | ")
        return f"TEXT seq={v[1]} pkt {v[3] + 1}/{v[2]} status=0x{v[4]:02x} page {v[7]}/{v[8]} \"{text[:60]}\""
    if c == 0x18:
        return "EXIT (back to dashboard)"
    if c == 0x0E:
        return f"MIC {'on' if len(v) > 1 and v[1] else 'off'}"
    if c == 0xF5:
        return f"EVENT {v[1] if len(v) > 1 else '?'} (TouchBar/head)"
    if c == 0x06:
        return f"DASHBOARD sub {v[4] if len(v) > 4 else '?'}"
    if c == 0x1E:
        return "QUICK NOTE"
    if c == 0x4B:
        return "NOTIFICATION"
    if c in (0x15, 0x16, 0x20):
        return "PICTURE (bmp)"
    if c == 0x25:
        return "heartbeat"
    return ""


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    show_all = "--all" in args
    only = None
    rest = [a for a in args if a != "--all"]
    if "--cmd" in rest:
        i = rest.index("--cmd")
        only = int(rest[i + 1], 16)
        rest = rest[:i] + rest[i + 2:]
    path = rest[0]
    data = read_log(path)
    arms = {}
    count = 0
    t0 = None
    for when, sent, handle, pdu, peers in att_packets(data):
        op = pdu[0]
        if op not in (WRITE_CMD, WRITE_REQ, NOTIFY) or len(pdu) < 3:
            continue
        value = pdu[3:]
        if not value:
            continue
        if value[0] == 0x25 and not show_all:
            continue
        if only is not None and value[0] != only:
            continue
        arm = arms.setdefault(handle, f"glasses#{len(arms) + 1} ({peers.get(handle, '?')})")
        t0 = t0 or when
        arrow = "phone -> " if op != NOTIFY else "phone <- "
        hexs = " ".join(f"{b:02x}" for b in value[:24]) + (" ..." if len(value) > 24 else "")
        print(f"{when.strftime('%H:%M:%S.%f')[:-3]}  {arrow}{arm}  {hexs}\n{'':14}{describe(value)}")
        count += 1
    if count == 0:
        print("Nothing to or from the glasses in this log (was Bluetooth restarted after turning the log on?)")
    else:
        print(f"\n{count} packets. Arms: " + ", ".join(arms.values()))


if __name__ == "__main__":
    main()
