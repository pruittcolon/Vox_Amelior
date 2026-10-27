"""End-to-end tests for the Agent Hub API the Even Suite app uses.

Runs the real server against a private tmux server and a fake `claude`
binary, so nothing touches the real sessions:  python3 -m unittest test_server
"""
import glob
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

FAKE_CLAUDE = os.path.join(HERE, "fake_claude.py")


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
            shutil.copy(FAKE_CLAUDE, path)
            os.chmod(path, 0o755)
        cls.spawn_log = os.path.join(cls.home, "claude-starts.jsonl")
        cls.port = free_port()
        env = {k: v for k, v in os.environ.items() if k != "TMUX" and not k.startswith("CLAUDE")}
        env.update(HOME=cls.home, HUB_PORT=str(cls.port), HUB_NO_BROWSER="1",
                   TMUX_TMPDIR=cls.home, HUB_WORKDIR=cls.home, PATH=bindir + os.pathsep + env.get("PATH", ""),
                   FAKE_CLAUDE_LOG=cls.spawn_log)
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
        self.assertEqual(data["api"], 3)
        self.assertIn("live", data["features"])
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

    def test_log_upload_and_summary(self):
        now = int(time.time() * 1000)
        events = [
            {"t": now, "c": "ble", "e": "req", "cmd": "4e", "lr": "L", "ms": 38, "ok": True},
            {"t": now + 10, "c": "bmp", "e": "frame", "ok": True, "stage": "done", "ms": 640,
             "packetsMs": 520, "endMs": 60, "bytes": 9854},
            {"t": now + 20, "c": "voice", "e": "final", "text": "clod what time is it", "mode": "waiting",
             "src": "phone"},
        ]
        r = self.post("/api/log", {"app": "even-suite", "session": "s1", "dropped": 0, "events": events})
        self.assertEqual(r["n"], 3)
        logs = glob.glob(os.path.join(self.home, ".agent-hub", "logs", "even-suite-*.jsonl"))
        self.assertEqual(len(logs), 1)
        with open(logs[0]) as f:
            lines = [json.loads(line) for line in f]
        self.assertEqual(lines[1]["packetsMs"], 520)
        self.assertEqual(lines[0]["session"], "s1")
        out = subprocess.run([sys.executable, os.path.join(HERE, "logstats.py"), logs[0]],
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("Picture frames: ok=1", out.stdout)
        self.assertIn("clod what time is it", out.stdout)

    # ---- live Claude (stream-json): stages, streaming text, queue, options, stop

    def starts(self):
        try:
            with open(self.spawn_log) as f:
                return [json.loads(line) for line in f if line.strip()]
        except OSError:
            return []

    def watch(self, text, timeout=25, options=None, seen=None):
        """Send text; return (final reply, every state of the reply seen on the way)."""
        before = {m["id"] for m in self.get("/api/bridge")["messages"]}
        body = {"text": text, "to": "claude"}
        if options is not None:
            body["options"] = options
        self.post("/api/bridge/send", body)
        version, deadline, states = -1, time.time() + timeout, []
        while time.time() < deadline:
            snap = self.get(f"/api/bridge?v={version}&wait=2", timeout=10)
            version = snap["version"]
            mine = [m for m in snap["messages"] if m["who"] == "claude" and m["id"] not in before]
            if mine:
                m = mine[-1]
                states.append({k: m.get(k) for k in ("status", "stage", "tool", "live", "steps", "block")})
                if m["status"] in ("done", "error"):
                    return m, states
        self.fail(f"no reply to {text!r}")

    def test_live_stream_shows_text_as_it_is_written(self):
        reply, states = self.watch("please stream this answer")
        self.assertEqual(reply["status"], "done")
        self.assertTrue(reply["text"].startswith("echo: please stream this answer"))
        partial = [s["live"] for s in states if s["status"] == "pending" and s["live"]]
        self.assertTrue(partial, "saw no partial text")
        self.assertTrue(any(len(p) < len(reply["text"]) for p in partial), partial)
        self.assertIn("writing", [s["stage"] for s in states])
        self.assertEqual(reply["live"], "")
        self.assertGreater(reply["ms"], 0)

    def test_live_tool_step_is_reported(self):
        reply, states = self.watch("use a tool please")
        self.assertEqual(reply["status"], "done")
        self.assertIn("tool", [s["stage"] for s in states])
        self.assertIn("List files", [s["tool"] for s in states])
        self.assertEqual(reply["steps"], ["List files"])

    def test_live_options_reuse_and_restart(self):
        opts = {"model": "opus", "effort": "medium", "access": "auto"}
        self.watch("first with options", options=opts)
        n = len(self.starts())
        args = self.starts()[-1]
        for flag, value in (("--model", "opus"), ("--effort", "medium"), ("--permission-mode", "auto"),
                            ("--permission-prompts", "none"), ("--input-format", "stream-json")):
            self.assertEqual(args[args.index(flag) + 1], value, flag)
        self.watch("second, same options", options=opts)
        self.assertEqual(len(self.starts()), n, "same options: the running Claude answers")
        self.watch("third, read only", options=dict(opts, access="read", model="bad model!"))
        args = self.starts()[-1]
        self.assertEqual(len(self.starts()), n + 1, "new options: a new Claude, same conversation")
        self.assertNotIn("--permission-mode", args)
        self.assertNotIn("--model", args, "model names are checked")
        self.assertIn("--resume", args)

    def test_live_queue_one_at_a_time(self):
        self.post("/api/bridge/send", {"text": "slow one", "to": "claude"})
        time.sleep(0.5)
        reply2, states2 = self.watch("quick two")
        self.assertEqual(reply2["status"], "done")
        self.assertIn("queued", [s["stage"] for s in states2], "the second question waited its turn")
        replies = [m for m in self.get("/api/bridge")["messages"] if m["who"] == "claude"]
        slow = [m for m in replies if m["text"].startswith("echo: slow one")]
        self.assertEqual(slow[-1]["status"], "done")

    def test_live_stop_then_continue(self):
        self.post("/api/bridge/send", {"text": "slow and stoppable", "to": "claude"})
        time.sleep(1.0)
        self.post("/api/bridge/stop", {})
        msgs = []
        deadline = time.time() + 10
        while time.time() < deadline:
            msgs = [m for m in self.get("/api/bridge")["messages"] if m["who"] == "claude"]
            if msgs and msgs[-1]["status"] == "error":
                break
            time.sleep(0.2)
        self.assertEqual(msgs[-1]["text"], "Stopped.")
        reply, _ = self.watch("after the stop")
        self.assertEqual(reply["status"], "done")
        self.assertIn("--resume", self.starts()[-1])

    def test_live_crash_is_an_error_not_a_hang(self):
        reply, _ = self.watch("please die now")
        self.assertEqual(reply["status"], "error")
        self.assertIn("died", reply["text"])
        reply, _ = self.watch("back again")
        self.assertEqual(reply["status"], "done")

    def test_warm_starts_claude_early(self):
        self.watch("make sure a session exists")
        self.post("/api/bridge/warm", {"options": {"effort": "low"}})
        deadline = time.time() + 5
        while time.time() < deadline and "low" not in json.dumps(self.starts()[-1:]):
            time.sleep(0.1)
        self.assertEqual(self.starts()[-1][self.starts()[-1].index("--effort") + 1], "low")

    # ---- Work tab: finished Claude Code jobs from the Stop hook

    def test_work_feed_long_poll(self):
        first = self.get("/api/work")
        t0 = time.time()
        same = self.get(f"/api/work?v={first['version']}&wait=1")
        self.assertGreaterEqual(time.time() - t0, 0.9)
        self.assertEqual(same["version"], first["version"])
        self.post("/api/work/done", {"project": "even-suite", "summary": "Pushed the fix.", "ms": 61000})
        now = self.get(f"/api/work?v={first['version']}&wait=5")
        self.assertEqual(now["items"][-1]["summary"], "Pushed the fix.")
        self.assertEqual(now["items"][-1]["ms"], 61000)

    def test_stop_hook_reports_the_last_answer(self):
        transcript = os.path.join(self.home, "t.jsonl")
        lines = [
            {"type": "user", "timestamp": "2026-10-27T10:00:00Z", "message": {"content": "build it"}},
            {"type": "assistant", "timestamp": "2026-10-27T10:00:30Z", "message": {"content": [
                {"type": "tool_use", "name": "Bash", "input": {}}]}},
            {"type": "user", "timestamp": "2026-10-27T10:00:40Z", "message": {"content": [
                {"type": "tool_result", "content": "ok"}]}},
            {"type": "assistant", "timestamp": "2026-10-27T10:02:00Z", "message": {"content": [
                {"type": "text", "text": "**Done.** The APK is built and CI is green. Next: install it."}]}},
        ]
        with open(transcript, "w") as f:
            f.write("\n".join(json.dumps(x) for x in lines))
        hook = os.path.join(HERE, "claude_done_hook.py")
        data = json.dumps({"transcript_path": transcript, "cwd": "/root/even-suite", "session_id": "abcdef123"})
        env = dict(self.env, HUB_PORT=str(self.port))
        r = subprocess.run([sys.executable, hook], input=data, text=True, env=env, timeout=20)
        self.assertEqual(r.returncode, 0)
        item = self.get("/api/work")["items"][-1]
        self.assertEqual(item["summary"], "Done.")
        self.assertEqual(item["project"], "even-suite")
        self.assertEqual(item["ms"], 120000)
        self.assertEqual(item["tools"], 1)
        before = self.get("/api/work")["version"]
        subprocess.run([sys.executable, hook], input=data, text=True, env=dict(env, HUB_BRIDGE="1"), timeout=20)
        self.assertEqual(self.get("/api/work")["version"], before, "the app's own questions are skipped")
        # Agent Hub not running: the job waits in a file.
        subprocess.run([sys.executable, hook], input=data, text=True, env=dict(env, HUB_PORT="1"), timeout=20)
        with open(os.path.join(self.home, ".agent-hub", "work-pending.jsonl")) as f:
            self.assertIn("Done.", f.read())

    def test_live_job_progress_then_done(self):
        hook = os.path.join(HERE, "claude_done_hook.py")
        env = dict(self.env, HUB_PORT=str(self.port))

        def run(d):
            r = subprocess.run([sys.executable, hook], input=json.dumps(d), text=True, env=env, timeout=20)
            self.assertEqual(r.returncode, 0)

        base = {"session_id": "job12345xyz", "cwd": "/root/proj"}
        v0 = self.get("/api/work")["version"]
        run(dict(base, hook_event_name="UserPromptSubmit", prompt="fix   the build"))
        run(dict(base, hook_event_name="PreToolUse", tool_name="Edit",
                 tool_input={"file_path": "/root/proj/server.py", "old_string": "x" * 5000}))
        snap = self.get(f"/api/work?v={v0}&wait=5")
        job = [j for j in snap["active"] if j["session"] == "job12345"][0]
        self.assertEqual(job["prompt"], "fix the build")
        self.assertEqual(job["step"], "Editing server.py")
        self.assertEqual(job["steps"], 1)
        self.assertEqual(job["project"], "proj")
        transcript = os.path.join(self.home, "job.jsonl")
        with open(transcript, "w") as f:
            f.write(json.dumps({"type": "user", "timestamp": "2026-10-27T10:00:00Z", "message": {"content": "fix the build"}}) + "\n")
            f.write(json.dumps({"type": "assistant", "timestamp": "2026-10-27T10:03:00Z",
                                "message": {"content": [{"type": "text", "text": "Fixed it."}]}}))
        run(dict(base, hook_event_name="Stop", transcript_path=transcript))
        snap = self.get("/api/work")
        self.assertFalse([j for j in snap["active"] if j["session"] == "job12345"], "done: no longer working")
        self.assertEqual(snap["items"][-1]["prompt"], "fix the build")
        self.assertEqual(snap["items"][-1]["summary"], "Fixed it.")

    def test_glasses_command_sends_a_message(self):
        script = os.path.join(HERE, "glasses")
        env = dict(self.env, HUB_PORT=str(self.port))
        r = subprocess.run([sys.executable, script, "Done: APK built.", "CI is green."], env=env,
                           capture_output=True, text=True, timeout=20, cwd=self.home)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("Sent to your glasses", r.stdout)
        item = self.get("/api/work")["items"][-1]
        self.assertEqual(item["kind"], "message")
        self.assertEqual(item["summary"], "Done: APK built.")
        self.assertEqual(item["text"], "Done: APK built. CI is green.")
        r = subprocess.run([sys.executable, script, "later"], env=dict(env, HUB_PORT="1"),
                           capture_output=True, text=True, timeout=20)
        self.assertIn("saved", r.stdout)

    def test_install_hook_is_idempotent(self):
        settings = os.path.join(self.home, ".claude", "settings.json")
        os.makedirs(os.path.dirname(settings), exist_ok=True)
        with open(settings, "w") as f:
            json.dump({"model": "opus", "hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo hi"}]}]}}, f)
        script = os.path.join(HERE, "install_hook.py")
        for _ in range(2):
            subprocess.run([sys.executable, script, "--quiet"], env=self.env, check=True, timeout=20)
        with open(settings) as f:
            data = json.load(f)
        self.assertEqual(data["model"], "opus")
        commands = [h["command"] for g in data["hooks"]["Stop"] for h in g["hooks"]]
        self.assertEqual(len(commands), 2)
        self.assertIn("echo hi", commands)
        self.assertTrue(any("claude_done_hook.py" in c for c in commands))
        for event in ("UserPromptSubmit", "PreToolUse"):
            [group] = data["hooks"][event]
            self.assertTrue(group["hooks"][0]["async"], f"{event} must not slow Claude down")
        command = os.path.join(self.home, ".claude", "commands", "even.md")
        with open(command) as f:
            self.assertIn("agent-hub/glasses", f.read(), "/even is installed with the hooks")
        self.assertTrue(self.get("/api/ping")["hook"])
        subprocess.run([sys.executable, script, "--remove", "--quiet"], env=self.env, check=True, timeout=20)
        self.assertFalse(self.get("/api/ping")["hook"])
        self.assertTrue(self.post("/api/work/install", {})["hook"])
        subprocess.run([sys.executable, script, "--remove", "--quiet"], env=self.env, check=True, timeout=20)
        with open(settings) as f:
            self.assertEqual([h["command"] for g in json.load(f)["hooks"]["Stop"] for h in g["hooks"]], ["echo hi"])
        self.assertFalse(os.path.exists(command), "/even goes with --remove")

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
