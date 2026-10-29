#!/usr/bin/env python3
"""Adds the Even Suite Work hooks (claude_done_hook.py) to
~/.claude/settings.json, once: Stop (Claude finished), and in the background
UserPromptSubmit (a job started) and PreToolUse (each step), so the glasses
can show a job live when you look up. Keeps everything else in the file as it
is and saves a backup the first time. Run again any time; it changes nothing
twice.

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
    hooks = settings.setdefault("hooks", {})

    def ours(event):
        return [g for g in hooks.get(event) or []
                if any("claude_done_hook.py" in h.get("command", "") for h in g.get("hooks", []))]

    wanted = {
        "Stop": {"hooks": [{"type": "command", "command": COMMAND, "timeout": 10}]},
        "UserPromptSubmit": {"hooks": [{"type": "command", "command": COMMAND, "timeout": 5, "async": True}]},
        "PreToolUse": {"hooks": [{"type": "command", "command": COMMAND, "timeout": 5, "async": True}]},
    }
    if remove:
        if not any(ours(e) for e in wanted):
            return 0
        for event in wanted:
            mine = ours(event)
            if mine:
                hooks[event] = [g for g in hooks.get(event) or [] if g not in mine]
                if not hooks[event]:
                    del hooks[event]
    else:
        if all(ours(e) == [g] for e, g in wanted.items()):
            command = os.path.join(os.path.dirname(SETTINGS), "commands", "even.md")
            os.makedirs(os.path.dirname(command), exist_ok=True)
            shutil.copy(os.path.join(HERE, "even-command.md"), command)
            if not quiet:
                print("The Even Suite Work hooks are already installed (and /even is up to date).")
            return 0
        for event, group in wanted.items():
            hooks[event] = [g for g in hooks.get(event) or [] if g not in ours(event)] + [group]
    # The /even command: glasses mode for a Claude Code conversation.
    commands = os.path.join(os.path.dirname(SETTINGS), "commands")
    command_file = os.path.join(commands, "even.md")
    if remove:
        if os.path.exists(command_file):
            os.remove(command_file)
    else:
        os.makedirs(commands, exist_ok=True)
        shutil.copy(os.path.join(HERE, "even-command.md"), command_file)
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
        print("Removed the hooks and /even." if remove else
              "Installed: Claude Code now tells Even Suite what it is doing and when it is done. "
              "In any Claude Code chat, type /even to get updates on your glasses.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
