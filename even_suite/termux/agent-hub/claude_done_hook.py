#!/usr/bin/env python3
"""Claude Code hook for the Even Suite Work tab and look-up screen.

- UserPromptSubmit: a job started (with what you asked);
- PreToolUse: the step Claude is on ("Editing server.py");
- Stop: Claude finished - with a one-line summary.

Agent Hub collects these; the app shows the job live when you look up and
says so (briefly) on the phone and glasses when it is done. Installed by
install_hook.py into ~/.claude/settings.json (start and step hooks run in the
background). Standard library only; never blocks Claude (short timeouts,
always exits 0). A finished job that finds Agent Hub not running is kept in
~/.agent-hub/work-pending.jsonl and shows up the next time it starts.
"""
import json
import os
import re
import sys
import time
import urllib.request
from datetime import datetime

PORT = os.environ.get("HUB_PORT", "8787")
PENDING = os.path.expanduser("~/.agent-hub/work-pending.jsonl")


def parse_ts(s):
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def is_prompt(entry):
    """A message the user typed (not a tool result handed back to Claude)."""
    if entry.get("type") != "user" or entry.get("isMeta"):
        return False
    content = (entry.get("message") or {}).get("content")
    if isinstance(content, str):
        return not content.startswith("<")
    return isinstance(content, list) and any(c.get("type") == "text" for c in content)


def last_turn(path):
    """(final text, seconds the turn took, tool calls) from the transcript."""
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            entries = [json.loads(line) for line in f if line.strip()]
    except (OSError, ValueError):
        return "", 0, 0
    start = 0
    for i, e in enumerate(entries):
        if is_prompt(e):
            start = i
    text, tools = "", 0
    for e in entries[start:]:
        if e.get("type") != "assistant":
            continue
        for c in (e.get("message") or {}).get("content") or []:
            if c.get("type") == "tool_use":
                tools += 1
            elif c.get("type") == "text" and c.get("text", "").strip():
                text = c["text"].strip()
    t0 = parse_ts(entries[start].get("timestamp")) if entries else None
    t1 = parse_ts(entries[-1].get("timestamp")) if entries else None
    secs = (t1 - t0) if t0 and t1 else 0
    return text, max(0, secs), tools


def summarize(text, n=90):
    """First sentence, without Markdown, at most n characters."""
    t = re.sub(r"```.*?```", " ", text, flags=re.S)
    t = re.sub(r"[*_`#>|]", "", t)
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
    t = " ".join(t.split())
    m = re.match(r"(.+?[.!?])(\s|$)", t)
    first = m.group(1) if m else t
    return first if len(first) <= n else first[:n - 3].rstrip() + "..."


def post(path, item, timeout):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(item).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    urllib.request.urlopen(req, timeout=timeout).read()


def progress(data, event):
    """Start of a job, or a step: only worth sending while Agent Hub is up."""
    cwd = data.get("cwd") or os.getcwd()
    item = {"session": (data.get("session_id") or "")[:8],
            "project": os.path.basename(cwd.rstrip("/")) or cwd}
    if event == "UserPromptSubmit":
        prompt = " ".join(str(data.get("prompt") or "").split())
        item.update(state="start", prompt=prompt[:200])
    else:
        inp = data.get("tool_input") if isinstance(data.get("tool_input"), dict) else {}
        keep = ("description", "command", "file_path", "notebook_path", "pattern", "query", "url")
        item.update(state="step", tool=data.get("tool_name") or "",
                    input={k: str(inp[k])[:120] for k in keep if k in inp})
    try:
        post("/api/work/progress", item, 1.0)
    except OSError:
        pass


def main():
    if os.environ.get("HUB_BRIDGE") == "1":
        return  # a question from the app itself: it shows the answer already
    try:
        data = json.load(sys.stdin)
    except ValueError:
        return
    event = data.get("hook_event_name") or "Stop"
    if event in ("UserPromptSubmit", "PreToolUse"):
        progress(data, event)
        return
    text, secs, tools = last_turn(data.get("transcript_path") or "")
    text = (data.get("last_assistant_message") or text or "").strip()
    cwd = data.get("cwd") or os.getcwd()
    item = {
        "ts": time.time(),
        "project": os.path.basename(cwd.rstrip("/")) or cwd,
        "where": os.environ.get("TMUX_PANE", "") and "tmux",
        "session": (data.get("session_id") or "")[:8],
        "summary": summarize(text) if text else "Finished.",
        "text": text[:6000],
        "ms": int(secs * 1000),
        "tools": tools,
    }
    try:
        post("/api/work/done", item, 1.5)
    except OSError:
        try:
            os.makedirs(os.path.dirname(PENDING), exist_ok=True)
            with open(PENDING, "a") as f:
                f.write(json.dumps(item) + "\n")
        except OSError:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001 - a hook must never get in Claude's way
        pass
    sys.exit(0)
