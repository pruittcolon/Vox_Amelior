#!/usr/bin/env python3
"""Summarize Even Suite timing logs (~/.agent-hub/logs/even-suite-*.jsonl).

    python3 logstats.py              # today's log
    python3 logstats.py --all        # every log file
    python3 logstats.py FILE...      # specific files
    python3 logstats.py --since 30   # only the last 30 minutes

Prints what matters for tuning when and how the app sends: Bluetooth round
trips per command, text-screen and picture-frame times (with phases), screen
mirroring rate, flicker runs, recognizer errors and what it heard, and Claude
reply times.
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
        by_cfg = defaultdict(list)
        for e in frames:
            by_cfg[(e.get("chunk", 194), e.get("gapMs"))].append(e)
        for (chunk, gap), es in sorted(by_cfg.items(), key=lambda kv: str(kv[0])):
            oks = [e["ms"] for e in es if e.get("ok")]
            print(f"  chunk={chunk} gap={gap}ms: ok {len(oks)}/{len(es)}  {fmt_ms(oks)}")
        shown = [e for e in frames if e.get("shown")]
        if any("shown" in e for e in frames):
            print(f"  displayed (both arms accepted): {len(shown)}/{len(frames)}")
        for arm in ("L", "R"):
            steps = Counter(e.get("steps" + arm) for e in frames if e.get("steps" + arm))
            if steps:
                print(f"  {arm} steps:", dict(steps.most_common(6)))
        fastf = [e for e in frames if e.get("fast")]
        if fastf:
            hits = sum(1 for e in fastf if e.get("viaPrevious"))
            print(f"  fast mode: {hits}/{len(fastf)} frames verified via the previous picture "
                  f"(one transfer), median {pct([e['ms'] for e in fastf], 50)} ms")
        normal = [e for e in frames if not e.get("fast") and e.get("ok")]
        if normal:
            print(f"  normal (send twice): median {pct([e['ms'] for e in normal], 50)} ms")
        for name in ("fast-off", "fallback", "clear", "reconfigure"):
            n = [e for e in ev if e.get("c") == "screen" and e.get("e") == name]
            if n:
                print(f"  screen.{name}: {len(n)}")
        retried = Counter(e.get("retried") for e in frames if e.get("retried"))
        if retried:
            print("  resent to arm after a rejection:", dict(retried))
        paces = [e for e in ev if e.get("c") == "bmp" and e.get("e") == "pace"]
        if paces:
            print("  pace changes:", [f"{e['arm']}:{e['from']}->{e['to']}" for e in paces][-12:])
        modes = [e for e in ev if e.get("c") == "bmp" and e.get("e") == "mode"]
        if modes:
            print("  switched to left-then-right:", len(modes), "times")
        last = [e for e in frames if e.get("gapL") is not None][-1:]
        if last:
            print(f"  latest pace: L {last[0]['gapL']} ms, R {last[0]['gapR']} ms, mode {last[0].get('mode')}")
        crc = Counter(e.get("crc") for e in frames if e.get("crc"))
        if crc:
            print("  CRC answers (L/R, c9 = match, ca = mismatch):", dict(crc.most_common(6)))
        sizes = Counter(e.get("bytes") for e in frames)
        print("  picture sizes:", dict(sizes))
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
        radio = {}
        waits = defaultdict(list)
        mic = Counter()
        for e in nat:
            for arm, st in (e.get("ble") or {}).items():
                for k in ("writes", "bytes", "busy", "dropped"):
                    arms[arm][k] += st.get(k, 0)
                if st.get("writes"):
                    waits[arm].append(st.get("waitMaxMs", 0))
                if st.get("phy"):
                    radio[arm] = f"{st.get('phy')} PHY, MTU {st.get('mtu')}"
            for k, v in (e.get("mic") or {}).items():
                if isinstance(v, (int, float)) and k != "decodeAvgMs":
                    mic[k] += v
        if arms:
            print("\nNative Bluetooth writes:")
            for arm in sorted(arms):
                a = arms[arm]
                print(f"  {arm}: writes={a['writes']} bytes={a['bytes']} busy-retries={a['busy']} "
                      f"dropped={a['dropped']} worst wait={max(waits[arm], default=0):.1f} ms  "
                      f"{radio.get(arm, '')}")
        if mic.get("packets"):
            print(f"Glasses mic: packets={mic['packets']} pcm bytes={mic['bytesOut']} "
                  f"pipe errors={mic['writeErrors']}")

    # Picture test (Diagnostics): every answer in order, per trial
    starts = [i for i, e in enumerate(ev) if e.get("c") == "lab" and e.get("e") == "start"]
    if starts:
        run = ev[starts[-1]:]
        print("\nPicture test (latest run): send = transfer result, crc = c9 accepted / ca rejected")
        trial = None
        line = []
        for e in run:
            if e.get("c") != "lab" or e.get("e") != "step":
                if e.get("c") == "lab" and e.get("e") == "done":
                    break
                continue
            if e["trial"] != trial:
                if line:
                    print("  " + " | ".join(line))
                trial = e["trial"]
                print(f" {trial}")
                line = []
            if e["arm"] == "L" or "arm at a time" in trial:
                line.append(f"{e['step']}:")
            line[-1] += f" {e['arm']}={e['crc']}"
        if line:
            print("  " + " | ".join(line))

    # Flicker (top/bottom spots): latest run, then the one-off tests
    fl = [e for e in ev if e.get("c") == "flicker"]
    fstarts = [i for i, e in enumerate(fl) if e.get("e") == "start"]
    if fl:
        print("\nFlicker:")
        if fstarts:
            run = fl[fstarts[-1]:]
            cfg = run[0]
            print(f"  latest run: top {cfg.get('topMs')} ms at height {cfg.get('topH')}, bottom {cfg.get('bottomMs')} ms "
                  f"at height {cfg.get('bottomH')}, depth {cfg.get('depth')}, switch={cfg.get('what')} "
                  f"method={cfg.get('method', 'state0 (old)')} clear={cfg.get('clear', 'no (old)')} "
                  f"clear wait={cfg.get('clearMs', '-')} ms move wait={cfg.get('settleMs', 0)} ms "
                  f"preview={cfg.get('previewMs', '-')} ms wait={cfg.get('wait')} moves to={cfg.get('arms')}")
            steps = [e for e in run if e.get("e") == "step"]
            if steps:
                late = sum(1 for e in steps if e.get("late"))
                bad = sum(1 for e in steps if not e.get("ok", True))
                print(f"  steps={len(steps)} slow={late} failed={bad} "
                      f"move timeouts={sum(1 for e in steps if e.get('moveTimeout'))} "
                      f"text timeouts={sum(1 for e in steps if e.get('textTimeout'))}")
                for side in ("top", "bottom"):
                    ss = [e for e in steps if e.get("side") == side]
                    if ss:
                        print(f"  {side:6} step {fmt_ms([e['ms'] for e in ss])} (asked {ss[-1].get('target')} ms)")
                clears = [e['clearMs'] for e in steps if e.get('clearMs')]
                if clears:
                    print(f"  clear  {fmt_ms(clears)}")
                print(f"  move   {fmt_ms([e['moveMs'] for e in steps if e.get('moveMs')])}")
                print(f"  text   {fmt_ms([e['textMs'] for e in steps if e.get('textMs')])}")
                replies = Counter(e["moveReply"] for e in steps if e.get("moveReply"))
                if replies:
                    print("  move answers:", ", ".join(f"{k} x{n}" for k, n in replies.most_common(5)))
            stats = [e for e in run if e.get("e") == "stats"]
            if stats:
                print("  switches/s by second:", " ".join(f"{e.get('rate')}" for e in stats[-30:]),
                      f"(asked {stats[-1].get('asked')})")
            for e in run:
                if e.get("e") == "silent":
                    print(f"  note: {e.get('note')} ({e.get('what')})")
                if e.get("e") == "change":
                    print(f"  changed while running: top {e.get('topMs')} ms h{e.get('topH')}, "
                          f"bottom {e.get('bottomMs')} ms h{e.get('bottomH')}, wait={e.get('wait')}, "
                          f"method={e.get('method')}, clear={e.get('clear')} {e.get('clearMs')} ms, "
                          f"move wait={e.get('settleMs')} ms, switch={e.get('what')}")
            stop = next((e for e in run if e.get("e") == "stop"), None)
            if stop:
                print(f"  stopped ({stop.get('reason')}): {stop.get('steps')} switches in {stop.get('secs')} s "
                      f"= {stop.get('rate')}/s of {stop.get('asked')}/s asked")
        tests = [e for e in fl if e.get("e") == "test"]
        for e in tests[-6:]:
            print(f"  by hand: {e.get('side')} h{e.get('h')} {e.get('ms')} ms (clear {e.get('clearMs', '-')} ms, "
                  f"move {e.get('moveMs')} ms "
                  f"{e.get('moveReply') or 'no answer'}, text {e.get('textMs')} ms {e.get('textReply') or 'no answer'})")
        for e in [e for e in fl if e.get("e") == "reset"][-3:]:
            print(f"  reset to height {e.get('homeH')} depth {e.get('homeD')}: ok={e.get('ok')} {e.get('ms')} ms")

    # Unsolicited packets from the glasses
    rx = [e for e in ev if e.get("c") == "ble" and e.get("e") == "rx"]
    if rx:
        kinds = Counter()
        first = {}
        for e in rx:
            kinds[e["kind"]] += e.get("count", 1)
            first.setdefault(e["kind"], e.get("first"))
        print("\nUnrequested packets from the glasses:")
        for k, n in kinds.most_common(10):
            print(f"  {k} x{n}  e.g. {first[k]}")

    # Glasses display confirmations (0xF5 0x12 seen right after accepted pictures)
    disp = [e for e in ev if e.get("c") == "glasses" and e.get("e") == "touch" and e.get("code") == 18]
    if disp:
        print(f"\nGlasses display events (0xF5 0x12): {len(disp)}")

    # App health
    errs = [e for e in ev if e.get("c") == "app" and e.get("e") in ("error", "crash")]
    jank = [e for e in ev if e.get("c") == "app" and e.get("e") == "jank"]
    life = [e for e in ev if e.get("c") == "app" and e.get("e") == "lifecycle"]
    if errs or jank or life:
        print(f"\nApp: errors={len(errs)} slow-UI reports={len(jank)} lifecycle changes={len(life)}")
        for e in errs[-5:]:
            print(f"  {e['e']}: {e.get('error', '')[:120]}")
        if jank:
            print(f"  worst UI frame: {max(e.get('worstMs', 0) for e in jank)} ms, "
                  f"slow frames: {sum(e.get('slowFrames', 0) for e in jank)}")

    # Link problems
    link = [e for e in ev if e.get("c") == "glasses" and e.get("e") in ("glassesDisconnected", "glassesConnected")]
    if link:
        print("\nGlasses link:", dict(Counter(f"{e['e']}({e.get('reason') or e.get('arm') or ''})" for e in link)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
