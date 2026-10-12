#!/usr/bin/env python3
"""Summarize Even Suite timing logs (~/.agent-hub/logs/even-suite-*.jsonl).

    python3 logstats.py              # today's log
    python3 logstats.py --all        # every log file
    python3 logstats.py FILE...      # specific files
    python3 logstats.py --since 30   # only the last 30 minutes

Prints what matters for tuning when and how the app sends: Bluetooth round
trips per command, text-screen and picture-frame times (with phases), screen
mirroring rate, recognizer errors and what it heard, and Claude reply times.
"""
import glob
import json
import os
import sys
import time
from collections import Counter, defaultdict

LOG_DIR = os.path.expanduser("~/.agent-hub/logs")


def pct(values, p):
    if not values:
        return None
    v = sorted(values)
    return v[min(len(v) - 1, int(round(p / 100 * (len(v) - 1))))]


def fmt_ms(values):
    if not values:
        return "-"
    return (f"n={len(values)} p50={pct(values, 50)} p90={pct(values, 90)} "
            f"max={max(values)} ms")


def load(paths, since_ms):
    events = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                if e.get("t", 0) >= since_ms:
                    events.append(e)
    events.sort(key=lambda e: e.get("t", 0))
    return events


def main(argv):
    since_ms = 0
    paths = []
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "--all":
            paths = sorted(glob.glob(os.path.join(LOG_DIR, "even-suite-*.jsonl")))
        elif a == "--since":
            since_ms = int((time.time() - float(args.pop(0)) * 60) * 1000)
        else:
            paths.append(a)
    if not paths:
        paths = [os.path.join(LOG_DIR, time.strftime("even-suite-%Y-%m-%d.jsonl"))]
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        print("No logs yet in", LOG_DIR)
        return 1
    ev = load(paths, since_ms)
    if not ev:
        print("No events in", ", ".join(paths))
        return 1
    span = (ev[-1]["t"] - ev[0]["t"]) / 1000
    print(f"{len(ev)} events over {span / 60:.1f} min from {', '.join(os.path.basename(p) for p in paths)}")
    sessions = {e.get("session") for e in ev}
    dropped = max((e.get("dropped", 0) for e in ev), default=0)
    print(f"app sessions: {len(sessions)}  dropped before upload: {dropped}")

    kinds = Counter(f"{e.get('c')}.{e.get('e')}" for e in ev)
    print("\nEvents:", ", ".join(f"{k}={n}" for k, n in kinds.most_common()))

    # Bluetooth round trips per command
    req = defaultdict(list)
    timeouts = Counter()
    for e in ev:
        if e.get("c") == "ble" and e.get("e") == "req":
            key = f"{e.get('cmd')}/{e.get('lr')}"
            if e.get("ok"):
                req[key].append(e.get("ms", 0))
            else:
                timeouts[key] += 1
    if req or timeouts:
        print("\nBluetooth request -> reply (cmd/arm):")
        for key in sorted(set(req) | set(timeouts)):
            print(f"  {key:8} {fmt_ms(req[key])}  timeouts={timeouts[key]}")
    batches = [e for e in ev if e.get("c") == "ble" and e.get("e") == "batch"]
    if batches:
        rates = [e["bytes"] / max(e["ms"], 1) for e in batches if e.get("ms")]
        print(f"  image batches: {fmt_ms([e['ms'] for e in batches])}  "
              f"~{pct(rates, 50) or 0:.1f} KB/s per arm (p50)")

    # Text screens
    scr = [e for e in ev if e.get("c") == "display" and e.get("e") == "screen"]
    if scr:
        ok = [e["ms"] for e in scr if e.get("ok")]
        bad = [e for e in scr if not e.get("ok")]
        by_mode = Counter(e.get("mode") for e in scr)
        print(f"\nText screens: {fmt_ms(ok)}  failed={len(bad)}  retries={sum(e.get('tries', 1) - 1 for e in scr)}")
        print("  by mode:", dict(by_mode))

    # Picture frames (images + screen mirroring)
    frames = [e for e in ev if e.get("c") == "bmp" and e.get("e") == "frame"]
    if frames:
        good = [e for e in frames if e.get("ok")]
        print(f"\nPicture frames: ok={len(good)} failed={len(frames) - len(good)}")
        print("  total   ", fmt_ms([e["ms"] for e in good]))
        print("  packets ", fmt_ms([e["packetsMs"] for e in good if e.get("packetsMs") is not None]))
        print("  end ack ", fmt_ms([e["endMs"] for e in good if e.get("endMs") is not None]))
        fails = Counter(e.get("stage") for e in frames if not e.get("ok"))
        if fails:
            print("  failed at:", dict(fails))

    # Screen mirroring
    cap = [e for e in ev if e.get("c") == "screen" and e.get("e") == "capture"]
    sent = [e for e in ev if e.get("c") == "screen" and e.get("e") == "sent" and e.get("ok")]
    if cap:
        print(f"\nScreen mirror: captured={len(cap)} sent={len(sent)}")
        print("  capture ", fmt_ms([e["captureMs"] for e in cap if e.get("captureMs") is not None]))
        print("  encode  ", fmt_ms([e["encodeMs"] for e in cap if e.get("encodeMs") is not None]))
        if len(sent) > 1:
            gaps = [(b["t"] - a["t"]) for a, b in zip(sent, sent[1:]) if b["t"] - a["t"] < 10000]
            if gaps:
                print(f"  frame gap {fmt_ms(gaps)}  -> {1000 / pct(gaps, 50):.2f} fps (median)")
        busy = sum(1 for e in cap if e.get("busy"))
        print(f"  frames captured while a send was still running: {busy}")
        intervals = [e for e in ev if e.get("c") == "screen" and e.get("e") == "interval"]
        if intervals:
            print("  capture interval changes:", [e["ms"] for e in intervals][-10:])

    # Voice
    finals = [e for e in ev if e.get("c") == "voice" and e.get("e") == "final"]
    errors = [e for e in ev if e.get("c") == "voice" and e.get("e") == "error"]
    wakes = [e for e in ev if e.get("c") == "voice" and e.get("e") == "wake"]
    if finals or errors or wakes:
        print(f"\nVoice: finals={len(finals)} wakes={len(wakes)} errors={len(errors)}")
        if errors:
            print("  errors:", dict(Counter(e.get("message") for e in errors)))
        src = Counter(e.get("src") for e in finals)
        print("  heard via:", dict(src))
        mics = [e for e in ev if e.get("c") == "voice" and e.get("e") == "source"]
        if mics:
            print("  glasses mic accepted by recognizer:", Counter(e.get("glassesMic") for e in mics))
        for e in finals[-8:]:
            print(f"  heard ({e.get('mode')},{e.get('src')}): {e.get('text', '')[:90]!r}")
        cmds = Counter(e.get("cmd") for e in ev if e.get("c") == "voice" and e.get("e") == "command")
        if cmds:
            print("  commands:", dict(cmds))

    # Claude
    replies = [e for e in ev if e.get("c") == "chat" and e.get("e") == "reply"]
    if replies:
        print(f"\nClaude replies: {fmt_ms([e['ms'] for e in replies if e.get('ms')])}  "
              f"chars p50={pct([e['chars'] for e in replies], 50)}  errors={sum(1 for e in replies if not e.get('ok'))}")

    # Native counters (summed)
    nat = [e for e in ev if e.get("c") == "native"]
    if nat:
        arms = defaultdict(lambda: Counter())
        waits = defaultdict(list)
        mic = Counter()
        for e in nat:
            for arm, st in (e.get("ble") or {}).items():
                for k in ("writes", "bytes", "busy", "dropped"):
                    arms[arm][k] += st.get(k, 0)
                if st.get("writes"):
                    waits[arm].append(st.get("waitMaxMs", 0))
            for k, v in (e.get("mic") or {}).items():
                if isinstance(v, (int, float)) and k != "decodeAvgMs":
                    mic[k] += v
        if arms:
            print("\nNative Bluetooth writes:")
            for arm in sorted(arms):
                a = arms[arm]
                print(f"  {arm}: writes={a['writes']} bytes={a['bytes']} busy-retries={a['busy']} "
                      f"dropped={a['dropped']} worst wait={max(waits[arm], default=0):.1f} ms")
        if mic.get("packets"):
            print(f"Glasses mic: packets={mic['packets']} pcm bytes={mic['bytesOut']} "
                  f"pipe errors={mic['writeErrors']}")

    # Link problems
    link = [e for e in ev if e.get("c") == "glasses" and e.get("e") in ("glassesDisconnected", "glassesConnected")]
    if link:
        print("\nGlasses link:", dict(Counter(f"{e['e']}({e.get('reason') or e.get('arm') or ''})" for e in link)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
