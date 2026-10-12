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
import select
import shutil
import signal
import struct
import subprocess
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


class Bridge:
    def __init__(self):
        self.cond = threading.Condition()
        self.version = 0
        self.messages = []
        self.convo = {"claude": None, "agy": None}  # conversation ids
        self.busy = {}                              # agent -> Popen
        self.relay_token = 0
        self.load()

    # ---- persistence & change notification
    def load(self):
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            self.messages = data.get("messages", [])
            self.convo.update(data.get("convo", {}))
            for m in self.messages:  # anything mid-flight died with the old server
                if m.get("status") == "pending":
                    m["status"], m["text"] = "error", "Interrupted (server restarted)."
        except (OSError, ValueError):
            pass

    def changed(self):
        with self.cond:
            self.version += 1
            os.makedirs(DATA_DIR, exist_ok=True)
            tmp = STATE_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"messages": self.messages[-500:], "convo": self.convo}, f)
            os.replace(tmp, STATE_FILE)
            self.cond.notify_all()

    def snapshot(self, since, wait, tail=0):
        with self.cond:
            if self.version == since and wait:
                self.cond.wait(timeout=min(wait, 30))
            messages = self.messages[-tail:] if tail > 0 else self.messages
            return {"version": self.version, "messages": messages,
                    "busy": sorted(self.busy)}

    def add(self, **msg):
        msg.update(id=uuid.uuid4().hex[:10], ts=time.time())
        with self.cond:
            self.messages.append(msg)
        self.changed()
        return msg

    # ---- talking to agents
    def run_agent(self, agent, prompt):
        """Blocking: send prompt to agent's bridge conversation, return reply."""
        first = self.convo.get(agent) is None
        if first:
            prompt = INTRO + prompt
        if agent == "claude":
            args = [CLAUDE_BIN, "-p", "--output-format", "json"]
            if first:
                self.convo["claude"] = str(uuid.uuid4())
                args += ["--session-id", self.convo["claude"]]
            else:
                args += ["--resume", self.convo["claude"]]
            proc = subprocess.Popen(args, cwd=WORKDIR, stdin=subprocess.PIPE,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdin = prompt
        else:
            args = [AGY_BIN, "--output-format", "json"]
            if not first:
                args += ["--conversation", self.convo["agy"]]
            args.append("--print=" + prompt)
            proc = subprocess.Popen(args, cwd=WORKDIR, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            stdin = None
        with self.cond:
            self.busy[agent] = proc
        self.changed()
        try:
            out, err = proc.communicate(stdin)
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
        if agent == "claude":
            if data.get("is_error"):
                raise RuntimeError(data.get("result") or "Claude returned an error")
            return (data.get("result") or "").strip()
        if data.get("conversation_id"):
            self.convo["agy"] = data["conversation_id"]
        if data.get("status") not in (None, "SUCCESS"):
            raise RuntimeError(data.get("response") or data.get("status"))
        return (data.get("response") or "").strip()

    def ask(self, agent, prompt, src):
        """Post a pending bubble for agent, fill it with the reply."""
        msg = self.add(who=agent, status="pending", text="", src=src)
        try:
            msg["text"], msg["status"] = self.run_agent(agent, prompt), "done"
        except Exception as e:  # noqa: BLE001 - shown to the user
            msg["text"], msg["status"] = str(e), "error"
        self.changed()
        return msg

    def send(self, text, to, turns):
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

    def stop(self):
        with self.cond:
            self.relay_token += 1
            procs = list(self.busy.values())
        for p in procs:
            p.terminate()

    def reset(self):
        self.stop()
        with self.cond:
            self.messages, self.convo = [], {"claude": None, "agy": None}
        self.changed()


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
            return self.send_json({"ok": True, "name": "agent-hub", "api": 2,
                                   "sessions": sorted(SESSIONS)})
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
            bridge.send(text, to, max(0, min(int(d.get("turns", 0)), 10)))
            return self.send_json({"ok": True})
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
