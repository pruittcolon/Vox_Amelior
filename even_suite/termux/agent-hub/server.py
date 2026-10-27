#!/usr/bin/env python3
"""Agent Hub: Claude Code + Google Antigravity in one web page, plus a bridge.

Standard library only. Terminals are tmux sessions streamed to the browser
over Server-Sent Events; keystrokes come back as small POSTs. The bridge runs
each agent in print mode (claude -p / agy --print) with its own long-lived
conversation, so the two can pass messages back and forth.
"""
import base64
import fcntl
import hashlib
import json
import os
import pty
import queue
import re
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

HOST = "127.0.0.1"
PORT = int(os.environ.get("HUB_PORT", "8787"))
HERE = os.path.dirname(os.path.abspath(__file__))
STATIC = os.path.join(HERE, "static")
DATA_DIR = os.path.expanduser("~/.agent-hub")
STATE_FILE = os.path.join(DATA_DIR, "bridge.json")
LOG_DIR = os.path.join(DATA_DIR, "logs")  # Even Suite timing logs (one JSON object per line)
WORKDIR = os.environ.get("HUB_WORKDIR", os.path.expanduser("~"))

CLAUDE_BIN = shutil.which("claude", path=os.path.expanduser("~/.local/bin")) or "claude"
AGY_BIN = shutil.which("agy", path=os.path.expanduser("~/.local/bin")) or "agy"

# tmux session per agent. "claude" is the same session the Termux `claude`
# command attaches to, so the web page and the terminal share one Claude.
SESSIONS = {
    "claude": {"session": os.environ.get("HUB_CLAUDE_SESSION", "claude"), "cmd": CLAUDE_BIN},
    "agy": {"session": os.environ.get("HUB_AGY_SESSION", "agy"), "cmd": AGY_BIN},
    # Plain shell for the Even Suite phone app's Terminal tab.
    "shell": {"session": os.environ.get("HUB_SHELL_SESSION", "shell"), "cmd": "bash -l"},
}
NAMES = {"claude": "Claude Code", "agy": "Antigravity", "shell": "Shell"}

# Keys the Even Suite app may send to a session (tmux key names).
ALLOWED_KEYS = {"Enter", "Escape", "Up", "Down", "Left", "Right", "Tab", "BTab", "Space",
                "BSpace", "PageUp", "PageDown", "C-c", "C-d", "y", "n",
                "1", "2", "3", "4", "5", "6", "7", "8", "9"}

# Started from inside a Claude session, these would make child Claudes think
# they are nested. Drop them so every agent starts clean.
for key in list(os.environ):
    if key.startswith("CLAUDE") and key != "CLAUDE_CONFIG_DIR":
        del os.environ[key]
os.environ.setdefault("TERM", "xterm-256color")
os.environ.setdefault("LANG", "C.UTF-8")


# ---------------------------------------------------------------- tmux

def tmux(*args, input=None):
    return subprocess.run(["tmux", *args], input=input, capture_output=True, text=True)


def ensure_session(agent):
    name = SESSIONS[agent]["session"]
    if tmux("has-session", "-t", f"={name}").returncode != 0:
        # Fall back to a shell when the agent exits so the tab never goes dead.
        cmd = f"{SESSIONS[agent]['cmd']}; echo; echo '[{agent} exited - type {agent} to restart]'; exec bash -i"
        tmux("new-session", "-d", "-s", name, "-x", "100", "-y", "35", "-c", WORKDIR, cmd)
    # Whichever screen (phone browser or Termux) was used last sets the size.
    tmux("set-option", "-t", name, "window-size", "latest")
    return name


def paste_to_session(agent, text, enter):
    name = ensure_session(agent)
    buf = f"hub-{uuid.uuid4().hex[:8]}"
    tmux("load-buffer", "-b", buf, "-", input=text)
    tmux("paste-buffer", "-p", "-d", "-b", buf, "-t", name)
    if enter:
        time.sleep(0.15)
        tmux("send-keys", "-t", name, "Enter")


def capture_session(agent, lines):
    name = ensure_session(agent)
    out = tmux("capture-pane", "-p", "-J", "-t", name, "-S", f"-{lines}").stdout
    return out.rstrip() + "\n"


def screen_hash(text):
    return hashlib.sha1(text.encode("utf-8", "replace")).hexdigest()[:16]


def capture_wait(agent, lines, known, wait):
    """Screen text and its hash; waits up to `wait` s for it to differ from `known`.

    The phone app long-polls this, so an idle terminal costs one request per
    `wait` seconds instead of one per second.
    """
    name = ensure_session(agent)
    deadline = time.time() + max(0.0, min(wait, 25.0))
    while True:
        out = tmux("capture-pane", "-p", "-J", "-t", name, "-S", f"-{lines}").stdout
        text = out.rstrip() + "\n"
        h = screen_hash(text)
        if h != known or time.time() >= deadline:
            return text, h
        time.sleep(0.4)


def send_keys(agent, keys):
    name = ensure_session(agent)
    for key in keys:
        if key in ALLOWED_KEYS:
            tmux("send-keys", "-t", name, key)


# The G1 font is basically Latin text; Claude Code's UI is full of box drawing,
# arrows and dingbats. Swap the meaningful ones for ASCII and drop the rest.
GLASSES_MAP = str.maketrans({"●": "*", "⏺": "*", "⎿": ">", "❯": ">", "›": ">", "→": "->",
                             "←": "<-", "✓": "v", "✔": "v", "✗": "x", "✘": "x", "…": "...",
                             "•": "*", "·": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
                             "─": " ", "│": " ", " ": " "})


def glasses_text(agent, lines):
    """Last `lines` non-empty lines of the pane, cleaned up for the G1 display."""
    out = []
    for raw in capture_session(agent, 200).splitlines():
        line = raw.translate(GLASSES_MAP)
        line = "".join(c for c in line if not ("←" <= c <= "⯿" or c >= "\U0001f000"))
        line = " ".join(line.split())
        if line:
            out.append(line)
    return "\n".join(out[-lines:])


# ------------------------------------------------------- web terminals

class Pty:
    """One browser tab attached to a tmux session through a pseudo-terminal."""

    live = {}
    lock = threading.Lock()

    def __init__(self, agent, cols, rows):
        session = ensure_session(agent)
        self.id = uuid.uuid4().hex
        pid, fd = pty.fork()
        if pid == 0:
            env = dict(os.environ, TERM="xterm-256color")
            env.pop("TMUX", None)
            os.execvpe("tmux", ["tmux", "attach-session", "-t", f"={session}"], env)
        self.pid, self.fd = pid, fd
        self.resize(cols, rows)
        with Pty.lock:
            Pty.live[self.id] = self

    def resize(self, cols, rows):
        cols, rows = max(20, min(int(cols), 500)), max(5, min(int(rows), 300))
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    def write(self, data):
        while data:
            n = os.write(self.fd, data)
            data = data[n:]

    def close(self):
        with Pty.lock:
            Pty.live.pop(self.id, None)
        try:
            os.kill(self.pid, signal.SIGHUP)  # detaches the tmux client only
        except ProcessLookupError:
            pass
        try:
            os.close(self.fd)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except ChildProcessError:
            pass


# --------------------------------------------------------------- bridge

INTRO = (
    "[Agent Hub bridge] You are chatting through a web bridge that connects "
    "the user, Claude Code (Anthropic) and Google Antigravity (agy). Messages "
    "may come from the user or from the other agent; each one is labelled. "
    "Keep replies focused and readable on a phone screen.\n\n"
)

# How much Claude may do on this phone when the app asks. "auto" lets it run
# commands and edit files behind Claude Code's own safety checks; "read" only
# looks. Nobody can click "allow" in print mode, so anything that would ask is
# refused (--permission-prompts none).
ACCESS_MODES = {"read": None, "edit": "acceptEdits", "auto": "auto"}
EFFORTS = {"low", "medium", "high", "xhigh", "max"}
WORK_FILE = os.path.join(DATA_DIR, "work.json")
# Claude jobs that finished while Agent Hub was not running (claude_done_hook.py).
WORK_PENDING = os.path.join(DATA_DIR, "work-pending.jsonl")


def clean_options(d):
    """Model / effort / access sent by the app, checked."""
    d = d if isinstance(d, dict) else {}
    opts = {}
    model = str(d.get("model") or "").strip()
    if model and model != "default" and re.fullmatch(r"[A-Za-z0-9._\-\[\]]{1,60}", model):
        opts["model"] = model
    if d.get("effort") in EFFORTS:
        opts["effort"] = d["effort"]
    if d.get("access") in ACCESS_MODES:
        opts["access"] = d["access"]
    return opts


def describe_tool(name, inp):
    """A tool call as a few words for the glasses ("Reading server.py")."""
    inp = inp if isinstance(inp, dict) else {}

    def short(s, n=40):
        s = " ".join(str(s).split())
        return s if len(s) <= n else s[:n - 3] + "..."

    def base(p):
        p = str(p or "").rstrip("/")
        return os.path.basename(p) or p

    if name == "Bash":
        return short(inp["description"]) if inp.get("description") else "Running " + short(inp.get("command", ""), 32)
    if name == "Read":
        return "Reading " + base(inp.get("file_path"))
    if name in ("Edit", "MultiEdit", "NotebookEdit"):
        return "Editing " + base(inp.get("file_path") or inp.get("notebook_path"))
    if name == "Write":
        return "Writing " + base(inp.get("file_path"))
    if name == "Grep":
        return "Searching for " + short(inp.get("pattern", ""), 28)
    if name == "Glob":
        return "Looking for " + short(inp.get("pattern", ""), 30)
    if name == "WebSearch":
        return "Searching the web: " + short(inp.get("query", ""), 26)
    if name == "WebFetch":
        return "Reading " + (urlparse(str(inp.get("url", ""))).netloc or "a web page")
    if name in ("Task", "Agent"):
        return "Helper: " + short(inp.get("description") or "working", 34)
    if name == "TodoWrite":
        return "Planning"
    return "Using " + (name.split("__")[-1] if name else "a tool")


class LiveClaude:
    """One long-running `claude -p` in stream-json mode: a question does not
    wait for Claude Code to start, and the reply streams in as it is written
    (stage, tool steps and text), so the app can show what is happening."""

    def __init__(self, bridge):
        self.bridge = bridge
        self.proc = None
        self.key = None            # options the process was started with
        self.events = None         # parsed stdout lines; None = the process ended
        self.stderr = []
        self.stopped = False
        self.turn = threading.Lock()        # one question at a time
        self.spawn_lock = threading.Lock()

    @staticmethod
    def args(opts, system, session, resume):
        a = [CLAUDE_BIN, "-p", "--input-format", "stream-json", "--output-format", "stream-json",
             "--verbose", "--include-partial-messages"]
        if opts.get("model"):
            a += ["--model", opts["model"]]
        if opts.get("effort"):
            a += ["--effort", opts["effort"]]
        mode = ACCESS_MODES.get(opts.get("access", "read"))
        if mode:
            a += ["--permission-mode", mode]
        a += ["--permission-prompts", "none"]
        if system:
            a += ["--append-system-prompt", system]
        a += ["--resume", session] if resume else ["--session-id", session]
        return a

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def ensure(self, opts, system):
        """The Claude process for these options, started (or restarted) if needed."""
        with self.spawn_lock:
            key = json.dumps([opts, system], sort_keys=True)
            if self.alive() and key == self.key:
                return self.proc
            self.close()
            b = self.bridge
            session = b.convo.get("claude")
            resume = session is not None
            if not resume:
                session = b.convo["claude"] = str(uuid.uuid4())
            # HUB_BRIDGE: the "Claude is done" hook skips these (the app shows them already).
            env = dict(os.environ, HUB_BRIDGE="1")
            proc = subprocess.Popen(self.args(opts, system, session, resume), cwd=WORKDIR, env=env,
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            self.proc, self.key, self.events, self.stderr = proc, key, queue.Queue(), []
            threading.Thread(target=self._read, args=(proc, self.events), daemon=True).start()
            threading.Thread(target=self._read_err, args=(proc, self.stderr), daemon=True).start()
            return proc

    @staticmethod
    def _read(proc, events):
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                events.put(json.loads(line))
            except ValueError:
                pass
        events.put(None)

    @staticmethod
    def _read_err(proc, buf):
        for line in proc.stderr:
            buf.append(line)
            del buf[:-40]

    def err_tail(self):
        return "".join(self.stderr).strip()[-1500:]

    def close(self):
        p, self.proc = self.proc, None
        if p is not None and p.poll() is None:
            try:
                p.stdin.close()
            except OSError:
                pass
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()

    def warm(self, opts, system):
        """Start Claude now (the user started talking), unless it is answering."""
        if self.turn.locked():
            return
        try:
            self.ensure(opts, system)
        except OSError:
            pass

    def stop(self):
        self.stopped = True
        self.close()

    def run(self, prompt, msg, opts, system):
        """Ask one question; fills msg's stage/tool/live as the answer comes in."""
        self.stopped = False
        b = self.bridge
        for attempt in (1, 2):
            try:
                proc = self.ensure(opts, system)
            except OSError as e:
                raise RuntimeError(f"Could not start Claude Code: {e}")
            events = self.events
            with b.cond:
                b.busy["claude"] = proc
            try:
                proc.stdin.write(json.dumps({"type": "user",
                                             "message": {"role": "user", "content": prompt}}) + "\n")
                proc.stdin.flush()
                got_any, result = self._consume(events, msg)
            except (OSError, ValueError):
                got_any, result = False, None
            finally:
                with b.cond:
                    b.busy.pop("claude", None)
            if result is not None:
                return result
            err = self.err_tail()
            self.close()
            if self.stopped:
                raise RuntimeError("Stopped.")
            if attempt == 1 and "No conversation found" in err:
                b.convo["claude"] = None  # the old session is gone: start a new one
                continue
            if attempt == 1 and not got_any and opts.get("access") == "auto":
                # Auto mode not available on this account: answer without it.
                opts = dict(opts, access="read")
                msg["note"] = "Auto mode was not available, so Claude could only look this time."
                continue
            raise RuntimeError(err or "Claude Code stopped without answering.")
        raise RuntimeError("Claude Code stopped without answering.")

    def _consume(self, events, msg):
        """Read events until the turn's result. Returns (got anything, final text or None)."""
        b = self.bridge
        got_any = streamed = False
        last_touch = 0.0
        while True:
            ev = events.get()
            if ev is None:
                return got_any, None
            got_any = True
            t = ev.get("type")
            changed = False
            if ev.get("parent_tool_use_id"):
                pass  # a helper agent's own steps: the main one reports its progress
            elif t == "system":
                if msg["stage"] == "starting":
                    msg["stage"], changed = "thinking", True
            elif t == "stream_event":
                e = ev.get("event") or {}
                if e.get("type") == "content_block_start":
                    kind = (e.get("content_block") or {}).get("type")
                    if kind == "text":
                        msg["block"] += 1
                        msg["live"], msg["stage"] = "", "writing"
                    elif kind == "tool_use":
                        name = (e.get("content_block") or {}).get("name", "")
                        msg["stage"], msg["tool"] = "tool", describe_tool(name, {})
                    elif kind in ("thinking", "redacted_thinking") and msg["stage"] != "writing":
                        msg["stage"] = "thinking"
                    changed = True
                elif e.get("type") == "content_block_delta":
                    d = e.get("delta") or {}
                    if d.get("type") == "text_delta":
                        streamed = True
                        msg["live"] += d.get("text", "")
                        changed = True
            elif t == "assistant":
                for c in (ev.get("message") or {}).get("content") or []:
                    if c.get("type") == "tool_use":
                        desc = describe_tool(c.get("name", ""), c.get("input"))
                        msg["stage"], msg["tool"] = "tool", desc
                        msg["steps"] = (msg["steps"] + [desc])[-20:]
                        changed = True
                    elif c.get("type") == "text" and not streamed:
                        # Older Claude Code without partial messages: whole blocks.
                        msg["block"] += 1
                        msg["live"], msg["stage"] = c.get("text", ""), "writing"
                        changed = True
            elif t == "user":
                if msg["stage"] == "tool":
                    msg["stage"], changed = "thinking", True
            elif t == "result":
                if ev.get("is_error") or ev.get("subtype") not in (None, "success"):
                    raise ResultError(ev.get("result") or ev.get("subtype") or "Claude returned an error")
                return True, (ev.get("result") or msg["live"] or "").strip()
            now = time.time()
            if changed and now - last_touch > 0.15:
                last_touch = now
                b.touch()


class ResultError(RuntimeError):
    """Claude answered with an error (the process itself is fine)."""


HOOK_EVENTS = ("Stop", "UserPromptSubmit", "PreToolUse")


def hook_installed():
    """Are the Work hooks (claude_done_hook.py on start, each step and done) in Claude Code's settings?"""
    try:
        with open(os.path.expanduser("~/.claude/settings.json")) as f:
            hooks = json.load(f).get("hooks") or {}
    except (OSError, ValueError, AttributeError):
        return False
    return all(any("claude_done_hook.py" in h.get("command", "")
                   for g in hooks.get(event) or [] for h in g.get("hooks") or [])
               for event in HOOK_EVENTS)


# A Claude Code job not heard from for this long is no longer shown as working.
JOB_STALE_SECS = 30 * 60


class Bridge:
    def __init__(self):
        self.cond = threading.Condition()
        self.version = 0
        self.messages = []
        self.convo = {"claude": None, "agy": None}  # conversation ids
        self.busy = {}                              # agent -> Popen
        self.relay_token = 0
        # Extra system prompt for Claude (the Even Suite app sends the glasses'
        # screen size so answers come in screen-sized chunks).
        self.system = ""
        # Model / effort / access picked in the app.
        self.options = {}
        self.live = LiveClaude(self)
        self.locks = {"claude": self.live.turn, "agy": threading.Lock()}
        # Finished Claude Code jobs (from the Stop hook) for the app's Work tab.
        self.work = []
        self.work_version = 0
        # Claude Code jobs in progress (from the start / step hooks), by session.
        self.jobs = {}
        self.load()
        self.load_work()

    # ---- persistence & change notification
    def load(self):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.messages = data.get("messages", [])
            self.convo.update(data.get("convo", {}))
            self.system = data.get("system", "")
            self.options = clean_options(data.get("options"))
            for m in self.messages:  # anything mid-flight died with the old server
                if m.get("status") in ("pending", "queued"):
                    m["status"], m["text"] = "error", "Interrupted (server restarted)."
                    m["stage"] = "error"
        except (OSError, ValueError):
            pass

    def changed(self):
        with self.cond:
            self.version += 1
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"messages": self.messages[-500:], "convo": self.convo,
                           "system": self.system, "options": self.options}, f)
            os.replace(tmp, STATE_FILE)
            self.cond.notify_all()

    def touch(self):
        """Something changed that is not worth saving yet (a reply streaming in)."""
        with self.cond:
            self.version += 1
            self.cond.notify_all()

    def snapshot(self, since, wait, tail=0):
        with self.cond:
            if self.version == since and wait:
                self.cond.wait(timeout=min(wait, 30))
            messages = self.messages[-tail:] if tail > 0 else self.messages
            return {"version": self.version, "messages": messages,
                    "busy": sorted(self.busy), "options": self.options}

    def add(self, **msg):
        msg.update(id=uuid.uuid4().hex[:10], ts=time.time())
        with self.cond:
            self.messages.append(msg)
        self.changed()
        return msg

    # ---- talking to agents
    def run_agent(self, agent, prompt, msg):
        """Blocking: send prompt to agent's bridge conversation, return reply."""
        first = self.convo.get(agent) is None
        if first:
            prompt = INTRO + prompt
        if agent == "claude":
            try:
                return self.live.run(prompt, msg, dict(self.options), self.system)
            except ResultError:
                raise
            except RuntimeError:
                if first:
                    self.convo["claude"] = None
                raise
        args = [AGY_BIN, "--output-format", "json"]
        if not first:
            args += ["--conversation", self.convo["agy"]]
        args.append("--print=" + prompt)
        proc = subprocess.Popen(args, cwd=WORKDIR, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        with self.cond:
            self.busy[agent] = proc
        self.changed()
        try:
            out, err = proc.communicate()
        finally:
            with self.cond:
                self.busy.pop(agent, None)
        if proc.returncode and proc.returncode < 0:
            raise RuntimeError("Stopped.")
        try:
            data = json.loads(out.strip().splitlines()[-1])
        except (ValueError, IndexError):
            if first:
                self.convo[agent] = None
            raise RuntimeError((err or out or "No output").strip()[-1500:])
        if data.get("conversation_id"):
            self.convo["agy"] = data["conversation_id"]
        if data.get("status") not in (None, "SUCCESS"):
            raise RuntimeError(data.get("response") or data.get("status"))
        return (data.get("response") or "").strip()

    def ask(self, agent, prompt, src):
        """Post a bubble for agent (queued behind an answer in progress), fill it with the reply."""
        lock = self.locks[agent]
        waiting = lock.locked()
        msg = self.add(who=agent, status="queued" if waiting else "pending", text="", src=src,
                       stage="queued" if waiting else "starting", tool="", live="", block=0,
                       steps=[], started=0, ms=0, note="")
        with lock:
            msg["status"], msg["stage"], msg["started"] = "pending", "starting", time.time()
            self.touch()
            try:
                text = self.run_agent(agent, prompt, msg)
                msg["text"], msg["status"], msg["stage"] = text, "done", "done"
            except Exception as e:  # noqa: BLE001 - shown to the user
                msg["text"], msg["status"], msg["stage"] = str(e), "error", "error"
            msg["live"], msg["ms"] = "", int((time.time() - msg["started"]) * 1000)
            self.changed()
        return msg

    def send(self, text, to, turns, system=None, options=None):
        if system is not None:
            self.system = system[:4000]
        if options is not None:
            self.options = clean_options(options)
        with self.cond:
            self.relay_token += 1
            token = self.relay_token
        self.add(who="you", to=to, text=text, status="done")
        targets = ["claude", "agy"] if to == "both" else [to]

        def worker(agent):
            msg = self.ask(agent, f"[From the user]\n{text}", "you")
            other = "agy" if agent == "claude" else "claude"
            for _ in range(turns):
                if msg["status"] != "done" or token != self.relay_token:
                    return
                prompt = (f"[From {NAMES[agent]}, via the bridge]\n{msg['text']}\n\n"
                          f"(Reply to {NAMES[agent]} directly. The user is watching.)")
                msg = self.ask(other, prompt, agent)
                agent, other = other, agent

        for agent in targets:
            threading.Thread(target=worker, args=(agent,), daemon=True).start()

    def warm(self, system=None, options=None):
        if system is not None:
            self.system = system[:4000]
        if options is not None:
            self.options = clean_options(options)
        threading.Thread(target=self.live.warm, args=(dict(self.options), self.system), daemon=True).start()

    def stop(self):
        with self.cond:
            self.relay_token += 1
            procs = [p for a, p in self.busy.items() if a != "claude"]
        self.live.stop()
        for p in procs:
            p.terminate()

    def reset(self):
        self.stop()
        with self.cond:
            self.messages, self.convo = [], {"claude": None, "agy": None}
        self.changed()

    # ---- finished Claude Code jobs (the app's Work tab)
    def load_work(self):
        try:
            with open(WORK_FILE) as f:
                self.work = json.load(f).get("items", [])[-100:]
        except (OSError, ValueError, AttributeError):
            self.work = []
        try:
            with open(WORK_PENDING) as f:
                lines = f.readlines()
            os.remove(WORK_PENDING)
        except OSError:
            return
        for line in lines[-50:]:
            try:
                self.add_work(json.loads(line), save=False)
            except (ValueError, TypeError, AttributeError):
                pass
        self.save_work()

    def save_work(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = WORK_FILE + ".tmp"
        with self.cond:
            items = list(self.work)
        with open(tmp, "w") as f:
            json.dump({"items": items}, f)
        os.replace(tmp, WORK_FILE)

    def add_work(self, d, save=True):
        def num(v):
            try:
                return max(0, int(float(v)))
            except (TypeError, ValueError):
                return 0

        item = {"id": uuid.uuid4().hex[:10],
                "ts": float(d.get("ts") or time.time()),
                "project": str(d.get("project") or "")[:60],
                "where": str(d.get("where") or "")[:60],
                "session": str(d.get("session") or "")[:40],
                "summary": str(d.get("summary") or "")[:240],
                "text": str(d.get("text") or "")[:6000],
                "ms": num(d.get("ms")),
                "tools": num(d.get("tools")),
                # "job" (the Stop hook) or "message" (`glasses "..."` from a Claude Code session)
                "kind": "message" if d.get("kind") == "message" else "job"}
        with self.cond:
            job = self.jobs.pop(item["session"], None) if item["session"] else None
            item["prompt"] = (job or {}).get("prompt", "") or str(d.get("prompt") or "")[:200]
            self.work = (self.work + [item])[-100:]
            self.work_version += 1
            self.cond.notify_all()
        if save:
            self.save_work()
        return item

    def job_progress(self, d):
        """A Claude Code job started (state "start", with the prompt) or took a step."""
        session = str(d.get("session") or "")[:40]
        if not session:
            return None
        now = time.time()
        with self.cond:
            job = self.jobs.get(session)
            if d.get("state") == "start" or job is None:
                job = {"session": session, "project": str(d.get("project") or "")[:60],
                       "prompt": "", "step": "Thinking", "steps": 0, "started": now, "recent": []}
                self.jobs[session] = job
            if d.get("state") == "start":
                job["prompt"] = str(d.get("prompt") or "")[:200]
            else:
                job["step"] = describe_tool(str(d.get("tool") or ""), d.get("input"))
                job["steps"] += 1
                job["recent"] = (job.get("recent", []) + [job["step"]])[-4:]
            job["updated"] = now
            self.work_version += 1
            self.cond.notify_all()
            return job

    def active_jobs(self):
        now = time.time()
        for s in [s for s, j in self.jobs.items() if now - j.get("updated", 0) > JOB_STALE_SECS]:
            del self.jobs[s]
        return sorted(self.jobs.values(), key=lambda j: -j.get("updated", 0))

    def clear_work(self):
        with self.cond:
            self.work = []
            self.work_version += 1
            self.cond.notify_all()
        self.save_work()

    def work_snapshot(self, since, wait, tail=30):
        deadline = time.time() + max(0.0, min(wait, 30))
        with self.cond:
            while self.work_version == since:
                left = deadline - time.time()
                if left <= 0:
                    break
                self.cond.wait(timeout=left)
            return {"version": self.work_version, "items": self.work[-max(1, min(tail, 100)):],
                    "active": self.active_jobs()}


bridge = Bridge()


# ------------------------------------------------------------------ http

MIME = {".html": "text/html; charset=utf-8", ".js": "text/javascript", ".css": "text/css",
        ".svg": "image/svg+xml", ".png": "image/png", ".json": "application/json"}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        pass

    # ---- helpers
    def send_json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(n) if n else b""

    def json_body(self):
        try:
            return json.loads(self.body() or b"{}")
        except ValueError:
            return {}

    # ---- routes
    def do_GET(self):
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        parts = url.path.strip("/").split("/")
        if url.path == "/api/bridge":
            return self.send_json(bridge.snapshot(int(q.get("v", -1)), float(q.get("wait", 0)),
                                                  int(q.get("tail", 0))))
        if url.path == "/api/ping":
            return self.send_json({"ok": True, "name": "agent-hub", "api": 3, "hook": hook_installed(),
                                   "features": ["live", "warm", "work"], "sessions": sorted(SESSIONS)})
        if url.path == "/api/work":
            return self.send_json(bridge.work_snapshot(int(q.get("v", -1)), float(q.get("wait", 0)),
                                                       int(q.get("tail", 30))))
        if url.path == "/api/status":
            return self.send_json({a: tmux("has-session", "-t", f"={s['session']}").returncode == 0
                                   for a, s in SESSIONS.items()})
        if len(parts) == 4 and parts[:2] == ["api", "term"] and parts[2] in SESSIONS:
            if parts[3] == "stream":
                return self.stream(parts[2], q.get("cols", 80), q.get("rows", 24))
            if parts[3] == "capture":
                text, h = capture_wait(parts[2], max(1, min(int(q.get("lines", 60)), 2000)),
                                       q.get("h", ""), float(q.get("wait", 0)))
                return self.send_json({"text": text, "h": h})
            if parts[3] == "glasses":
                return self.send_json({"text": glasses_text(parts[2], int(q.get("lines", 20)))})
        return self.static(url.path)

    def do_POST(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        if parts[:2] == ["api", "pty"] and len(parts) == 4:
            p = Pty.live.get(parts[2])
            if not p:
                return self.send_json({"error": "gone"}, 410)
            try:
                if parts[3] == "input":
                    p.write(self.body())
                elif parts[3] == "resize":
                    d = self.json_body()
                    p.resize(d.get("cols", 80), d.get("rows", 24))
            except OSError:
                return self.send_json({"error": "gone"}, 410)
            return self.send_json({"ok": True})
        if parts[:2] == ["api", "term"] and len(parts) == 4 and parts[2] in SESSIONS and parts[3] == "paste":
            d = self.json_body()
            paste_to_session(parts[2], d.get("text", ""), bool(d.get("enter")))
            return self.send_json({"ok": True})
        if parts[:2] == ["api", "term"] and len(parts) == 4 and parts[2] in SESSIONS and parts[3] == "keys":
            keys = self.json_body().get("keys") or []
            if not isinstance(keys, list) or not all(isinstance(k, str) and k in ALLOWED_KEYS for k in keys):
                return self.send_json({"error": "bad keys"}, 400)
            send_keys(parts[2], keys)
            return self.send_json({"ok": True})
        if parts == ["api", "bridge", "send"]:
            d = self.json_body()
            text, to = (d.get("text") or "").strip(), d.get("to")
            if not text or to not in ("claude", "agy", "both"):
                return self.send_json({"error": "bad request"}, 400)
            system = d.get("system")
            bridge.send(text, to, max(0, min(int(d.get("turns", 0)), 10)),
                        system if isinstance(system, str) else None, d.get("options"))
            return self.send_json({"ok": True})
        if parts == ["api", "bridge", "warm"]:
            d = self.json_body()
            system = d.get("system")
            bridge.warm(system if isinstance(system, str) else None, d.get("options"))
            return self.send_json({"ok": True})
        if parts == ["api", "work", "done"]:
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "bad request"}, 400)
            return self.send_json({"ok": True, "id": bridge.add_work(d)["id"]})
        if parts == ["api", "work", "progress"]:
            d = self.json_body()
            if not isinstance(d, dict):
                return self.send_json({"error": "bad request"}, 400)
            return self.send_json({"ok": bridge.job_progress(d) is not None})
        if parts == ["api", "work", "clear"]:
            bridge.clear_work()
            return self.send_json({"ok": True})
        if parts == ["api", "work", "install"]:
            r = subprocess.run([sys.executable, os.path.join(HERE, "install_hook.py"), "--quiet"],
                               capture_output=True, text=True, timeout=30)
            return self.send_json({"ok": r.returncode == 0, "hook": hook_installed(),
                                   "error": (r.stdout + r.stderr).strip()[-500:]})
        if parts == ["api", "log"]:
            d = self.json_body()
            events = d.get("events")
            if not isinstance(events, list):
                return self.send_json({"error": "bad request"}, 400)
            os.makedirs(LOG_DIR, exist_ok=True)
            path = os.path.join(LOG_DIR, time.strftime("even-suite-%Y-%m-%d.jsonl"))
            meta = {"session": d.get("session", ""), "dropped": d.get("dropped", 0)}
            with open(path, "a") as f:
                for e in events[:5000]:
                    if isinstance(e, dict):
                        f.write(json.dumps({**e, **meta}, separators=(",", ":")) + "\n")
            return self.send_json({"ok": True, "n": len(events)})
        if parts == ["api", "quit"]:
            # `all stop` in Termux. Claude's tmux session keeps running.
            self.send_json({"ok": True})
            threading.Thread(target=lambda: (time.sleep(0.3), os.kill(os.getpid(), signal.SIGTERM)),
                             daemon=True).start()
            return
        if parts == ["api", "bridge", "stop"]:
            bridge.stop()
            return self.send_json({"ok": True})
        if parts == ["api", "bridge", "reset"]:
            bridge.reset()
            return self.send_json({"ok": True})
        self.send_json({"error": "not found"}, 404)

    def static(self, path):
        if path == "/":
            path = "/index.html"
        full = os.path.realpath(os.path.join(STATIC, path.lstrip("/")))
        if not full.startswith(STATIC + os.sep) or not os.path.isfile(full):
            return self.send_json({"error": "not found"}, 404)
        with open(full, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", MIME.get(os.path.splitext(full)[1], "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def stream(self, agent, cols, rows):
        p = Pty(agent, cols, rows)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            self.wfile.write(f"event: hello\ndata: {p.id}\n\n".encode())
            self.wfile.flush()
            while True:
                ready, _, _ = select.select([p.fd], [], [], 15)
                if not ready:
                    self.wfile.write(b": ping\n\n")  # notices closed tabs
                    self.wfile.flush()
                    continue
                try:
                    chunk = os.read(p.fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    self.wfile.write(b"event: bye\ndata: x\n\n")
                    break
                self.wfile.write(b"data: " + base64.b64encode(chunk) + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            p.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def open_browser(url):
    """Prefer Chrome; fall back to whatever Android opens links with."""
    tries = [["am", "start", "-a", "android.intent.action.VIEW", "-d", url, "-p", "com.android.chrome"],
             ["termux-open-url", url], ["xdg-open", url]]
    for cmd in tries:
        if shutil.which(cmd[0]):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and "Error" not in (r.stdout + r.stderr):
                return True
    return False


def main():
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    # Antigravity is big; it starts the first time someone opens it.
    for agent in ("claude", "shell"):
        ensure_session(agent)
    srv = Server((HOST, PORT), Handler)
    url = f"http://{HOST}:{PORT}/"
    print(f"  Agent Hub running at {url}")
    if os.environ.get("HUB_NO_BROWSER") != "1":
        threading.Thread(target=open_browser, args=(url,), daemon=True).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        for p in list(Pty.live.values()):
            p.close()


if __name__ == "__main__":
    main()
