#!/usr/bin/env python3
"""Adds the "Claude is done" Stop hook (claude_done_hook.py) to
~/.claude/settings.json, once. Keeps everything else in the file as it is and
saves a backup the first time. Run again any time; it changes nothing twice.

    python3 install_hook.py           # install
    python3 install_hook.py --remove  # take it out again
"""
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS = os.path.expanduser("~/.claude/settings.json")
HOOK = os.path.join(HERE, "claude_done_hook.py")
COMMAND = f"python3 {HOOK}"


def main():
    remove = "--remove" in sys.argv
    quiet = "--quiet" in sys.argv
    try:
        with open(SETTINGS) as f:
            settings = json.load(f)
    except FileNotFoundError:
        settings = {}
    except ValueError:
        print("~/.claude/settings.json is not valid JSON; not touching it.")
        return 1
    stop = settings.setdefault("hooks", {}).setdefault("Stop", [])
    ours = [g for g in stop if any("claude_done_hook.py" in h.get("command", "") for h in g.get("hooks", []))]
    if remove:
        if not ours:
            return 0
        settings["hooks"]["Stop"] = [g for g in stop if g not in ours]
    else:
        if ours and all(h.get("command") == COMMAND for g in ours for h in g.get("hooks", [])):
            if not quiet:
                print("The Claude-is-done hook is already installed.")
            return 0
        settings["hooks"]["Stop"] = [g for g in stop if g not in ours] + [
            {"hooks": [{"type": "command", "command": COMMAND, "timeout": 10}]}]
    os.makedirs(os.path.dirname(SETTINGS), exist_ok=True)
    backup = SETTINGS + ".bak-even-suite"
    if os.path.exists(SETTINGS) and not os.path.exists(backup):
        shutil.copy2(SETTINGS, backup)
    tmp = SETTINGS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")
    os.replace(tmp, SETTINGS)
    if not quiet:
        print("Removed the hook." if remove else "Installed: Claude Code now tells Even Suite when it finishes a job.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
