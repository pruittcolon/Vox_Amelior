#!/usr/bin/env python3
"""Stand-in for `claude` in tests: one-shot JSON (-p) or a stream-json chat.

Words in the message change what it does: "tool" runs a fake Bash step,
"slow" takes 3 s, "stream" sends its answer in slow pieces, "die" exits.
FAKE_CLAUDE_LOG (a file) records the arguments of every start.

Started in a terminal (a tmux session) it acts like Claude Code's screen: a
"❯ " input line, "● echo: ..." for each message, and for a message with
"ask" in it a menu ("❯ 1. Yes") that the next line answers.
"""
import json
import os
import sys
import time

args = sys.argv[1:]
if os.environ.get("FAKE_CLAUDE_LOG"):
    with open(os.environ["FAKE_CLAUDE_LOG"], "a") as f:
        f.write(json.dumps(args) + "\n")
mark = " [glasses]" if "--append-system-prompt" in args else ""


def reply_for(text):
    lines = text.strip().splitlines()
    return "echo: " + (lines[-1] if lines else "") + mark


if sys.stdin.isatty() and "-p" not in args and "--input-format" not in args:
    rule = "─" * 30
    while True:
        sys.stdout.write(f"{rule}\n❯ ")
        sys.stdout.flush()
        line = sys.stdin.readline()
        if not line:
            sys.exit(0)
        print(f"● echo: {line.strip()}")
        if "ask" in line:
            sys.stdout.write("Do you want to proceed?\n❯ 1. Yes\n  2. No\n")
            sys.stdout.flush()
            print(f"● answered: {sys.stdin.readline().strip()}")

if "--input-format" not in args:
    print(json.dumps({"result": reply_for(sys.stdin.read()), "is_error": False}))
    sys.exit(0)


def out(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def event(e):
    out({"type": "stream_event", "event": e, "parent_tool_use_id": None})


out({"type": "system", "subtype": "init", "session_id": "fake", "model": "fake"})
for line in sys.stdin:
    if not line.strip():
        continue
    content = json.loads(line)["message"]["content"]
    if not isinstance(content, str):
        content = " ".join(c.get("text", "") for c in content if c.get("type") == "text")
    last = content.strip().splitlines()[-1] if content.strip() else ""
    if "die" in last:
        sys.stderr.write("fake claude died\n")
        sys.exit(3)
    if "slow" in last:
        time.sleep(3)
    out({"type": "system", "subtype": "init", "session_id": "fake", "model": "fake"})
    if "tool" in last:
        event({"type": "content_block_start", "index": 0,
               "content_block": {"type": "tool_use", "id": "t1", "name": "Bash", "input": {}}})
        out({"type": "assistant", "parent_tool_use_id": None, "message": {"content": [
            {"type": "tool_use", "id": "t1", "name": "Bash",
             "input": {"command": "ls -la", "description": "List files"}}]}})
        time.sleep(1.0)
        out({"type": "user", "message": {"content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "a b"}]}})
    answer = reply_for(content)
    event({"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}})
    pause = 0.25 if "stream" in last else 0.01
    for i in range(0, len(answer), 6):
        event({"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": answer[i:i + 6]}})
        time.sleep(pause)
    event({"type": "content_block_stop", "index": 1})
    out({"type": "assistant", "parent_tool_use_id": None, "message": {"content": [{"type": "text", "text": answer}]}})
    out({"type": "result", "subtype": "success", "is_error": False, "result": answer, "duration_ms": 10})
