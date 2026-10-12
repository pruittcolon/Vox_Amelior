"""End-to-end tests for the Agent Hub API the Even Suite app uses.

Runs the real server against a private tmux server and a fake `claude`
binary, so nothing touches the real sessions:  python3 -m unittest test_server
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_CLAUDE = """#!/usr/bin/env python3
import json, sys
prompt = sys.stdin.read().strip().splitlines()
mark = " [glasses]" if "--append-system-prompt" in sys.argv else ""
print(json.dumps({"result": "echo: " + (prompt[-1] if prompt else "") + mark, "is_error": False}))
"""


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class AgentHubTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = tempfile.mkdtemp(prefix="hub-test-")
        bindir = os.path.join(cls.home, ".local", "bin")
        os.makedirs(bindir)
        for name in ("claude", "agy"):
            path = os.path.join(bindir, name)
            with open(path, "w") as f:
                f.write(FAKE_CLAUDE)
            os.chmod(path, 0o755)
        cls.port = free_port()
        env = {k: v for k, v in os.environ.items() if k != "TMUX" and not k.startswith("CLAUDE")}
        env.update(HOME=cls.home, HUB_PORT=str(cls.port), HUB_NO_BROWSER="1",
                   TMUX_TMPDIR=cls.home, HUB_WORKDIR=cls.home, PATH=bindir + os.pathsep + env.get("PATH", ""))
        cls.env = env
        cls.proc = subprocess.Popen([sys.executable, os.path.join(HERE, "server.py")], env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        cls.base = f"http://127.0.0.1:{cls.port}"
        for _ in range(100):
            try:
                cls.get("/api/ping")
                return
            except OSError:
                time.sleep(0.1)
        raise RuntimeError("server did not start")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        try:
            cls.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            cls.proc.kill()
        subprocess.run(["tmux", "kill-server"], env=cls.env, capture_output=True)
        shutil.rmtree(cls.home, ignore_errors=True)

    @classmethod
    def get(cls, path, timeout=10):
        with urllib.request.urlopen(cls.base + path, timeout=timeout) as r:
            return json.loads(r.read())

    @classmethod
    def post(cls, path, body):
        req = urllib.request.Request(cls.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def test_ping_lists_sessions(self):
        data = self.get("/api/ping")
        self.assertTrue(data["ok"])
        self.assertEqual(data["api"], 2)
        self.assertIn("shell", data["sessions"])
        status = self.get("/api/status")
        self.assertTrue(status["shell"])

    def test_bridge_send_and_tail(self):
        self.post("/api/bridge/send", {"text": "hello there", "to": "claude"})
        version, reply = -1, None
        deadline = time.time() + 20
        while time.time() < deadline and reply is None:
            snap = self.get(f"/api/bridge?v={version}&wait=5", timeout=15)
            version = snap["version"]
            for m in snap["messages"]:
                if m["who"] == "claude" and m["status"] == "done":
                    reply = m
        self.assertIsNotNone(reply, "no reply from fake claude")
        self.assertEqual(reply["text"], "echo: hello there")
        tail = self.get("/api/bridge?tail=1")
        self.assertEqual(len(tail["messages"]), 1)

    def wait_reply(self, after_id=None, timeout=20):
        version, deadline = -1, time.time() + timeout
        while time.time() < deadline:
            snap = self.get(f"/api/bridge?v={version}&wait=5", timeout=15)
            version = snap["version"]
            done = [m for m in snap["messages"] if m["who"] == "claude" and m["status"] == "done"]
            if done and done[-1]["id"] != after_id:
                return done[-1]
        return None

    def test_bridge_system_prompt(self):
        before = self.get("/api/bridge")["messages"]
        last = [m["id"] for m in before if m["who"] == "claude"]
        self.post("/api/bridge/send", {"text": "fit the glasses", "to": "claude",
                                       "system": "Answer in 200-character chunks."})
        reply = self.wait_reply(last[-1] if last else None)
        self.assertIsNotNone(reply)
        self.assertEqual(reply["text"], "echo: fit the glasses [glasses]")

    def test_zz_quit(self):
        # Last test: `all stop` shuts the server down cleanly.
        self.assertTrue(self.post("/api/quit", {})["ok"])
        self.proc.wait(timeout=10)
        self.assertIsNotNone(self.proc.returncode)

    def test_capture_long_poll_and_paste(self):
        first = self.get("/api/term/shell/capture?lines=50")
        self.assertIn("h", first)
        # Unchanged screen: the long poll waits, then returns the same hash.
        t0 = time.time()
        same = self.get(f"/api/term/shell/capture?lines=50&wait=1.2&h={first['h']}")
        if same["h"] == first["h"]:
            self.assertGreaterEqual(time.time() - t0, 1.0)
        self.post("/api/term/shell/paste", {"text": "echo even-suite-$((40+2))", "enter": True})
        deadline, seen = time.time() + 10, ""
        h = same["h"]
        while time.time() < deadline and "even-suite-42" not in seen:
            screen = self.get(f"/api/term/shell/capture?lines=50&wait=3&h={h}")
            seen, h = screen["text"], screen["h"]
        self.assertIn("even-suite-42", seen)

    def test_keys(self):
        self.assertTrue(self.post("/api/term/shell/keys", {"keys": ["Enter", "C-c"]})["ok"])
        with self.assertRaises(urllib.error.HTTPError) as cm:
            self.post("/api/term/shell/keys", {"keys": ["rm -rf /"]})
        self.assertEqual(cm.exception.code, 400)

    def test_glasses_text_is_plain(self):
        self.post("/api/term/shell/paste", {"text": "echo '● done → ok'", "enter": True})
        time.sleep(1)
        text = self.get("/api/term/shell/glasses?lines=10")["text"]
        self.assertNotIn("●", text)
        self.assertNotIn("→", text)


if __name__ == "__main__":
    unittest.main()
