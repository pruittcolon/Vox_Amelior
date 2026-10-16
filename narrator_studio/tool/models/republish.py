#!/usr/bin/env python3
"""Fetch the free voice models, unpack them, and republish them file by file.

The phone downloads single files (with resume and a sha256 check) from the
`narrator-models` release instead of unpacking big .tar.bz2 archives, which is
slow in Dart and needs twice the space.

  republish.py fetch   --work DIR            download + unpack every pack
  republish.py stage   --work DIR --out DIR  flatten files, zip folders, write manifest.json
  republish.py publish --out DIR             upload the staged files to the release
"""

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile

BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
RELEASE = "narrator-models"

# Files we never ship: Chinese text front-end data (the app reads English).
SKIP = ["*zh*", "dict", "dict/*", "*.fst", "*.far", "*.py", "*.sh", "*.md"]

PACKS = {
    "kokoro": {
        "archive": "kokoro-int8-multi-lang-v1_0.tar.bz2",
        "title": "Kokoro v1.0 (int8), 53 voices",
        "license": "Apache-2.0 (hexgrad/Kokoro-82M)",
    },
    "pocket": {
        "archive": "sherpa-onnx-pocket-tts-int8-2026-01-26.tar.bz2",
        "title": "Pocket TTS (int8), voice cloning",
        "license": "CC-BY-4.0 (kyutai/pocket-tts)",
    },
    "supertonic": {
        "archive": "sherpa-onnx-supertonic-3-tts-int8-2026-05-11.tar.bz2",
        "title": "Supertonic 3 (int8), 10 fast voices",
        "license": "OpenRAIL (Supertone/supertonic-3), code MIT",
    },
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pack_dir(work, name):
    archive = PACKS[name]["archive"]
    return os.path.join(work, archive[: -len(".tar.bz2")])


def fetch(work):
    os.makedirs(work, exist_ok=True)
    for name, pack in PACKS.items():
        target = pack_dir(work, name)
        if os.path.isdir(target):
            print(f"{name}: already unpacked")
            continue
        url = BASE + pack["archive"]
        dest = os.path.join(work, pack["archive"])
        print(f"{name}: downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dest)
        with tarfile.open(dest, "r:bz2") as tar:
            tar.extractall(work, filter="data")
        os.remove(dest)
        if not os.path.isdir(target):
            sys.exit(f"{name}: expected folder {target} after unpacking")
    for name in PACKS:
        root = pack_dir(work, name)
        print(f"\n== {name}: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames.sort()
            rel = os.path.relpath(dirpath, root)
            depth = 0 if rel == "." else rel.count(os.sep) + 1
            if depth > 1:
                continue
            for f in sorted(filenames):
                p = os.path.join(dirpath, f)
                print(f"  {os.path.relpath(p, root):60s} {os.path.getsize(p):>12,d}")
        for doc in ("README.md", "LICENSE", "LICENSE.txt", "README"):
            p = os.path.join(root, doc)
            if os.path.isfile(p):
                with open(p, encoding="utf-8", errors="replace") as f:
                    print(f"\n-- {name}/{doc} (first 60 lines)")
                    for i, line in enumerate(f):
                        if i >= 60:
                            break
                        print("   " + line.rstrip())


def skipped(rel):
    return any(fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(os.path.basename(rel), pat) for pat in SKIP)


def zip_folder(src, dest):
    with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for dirpath, dirnames, filenames in os.walk(src):
            dirnames.sort()
            for f in sorted(filenames):
                p = os.path.join(dirpath, f)
                z.write(p, os.path.relpath(p, os.path.dirname(src)))


def stage(work, out, gallery=None):
    os.makedirs(out, exist_ok=True)
    manifest = {"version": 1, "release": RELEASE, "packs": {}}
    for name, pack in PACKS.items():
        root = pack_dir(work, name)
        files = []
        for entry in sorted(os.listdir(root)):
            p = os.path.join(root, entry)
            if skipped(entry):
                print(f"{name}: skip {entry}")
                continue
            if os.path.isdir(p):
                if entry == "test_wavs":
                    # Reference voices: keep them as separate small files.
                    for wav in sorted(os.listdir(p)):
                        if wav.endswith(".wav"):
                            files.append((f"{name}--{wav}", f"test_wavs/{wav}", os.path.join(p, wav), False))
                    continue
                asset = f"{name}--{entry}.zip"
                zip_folder(p, os.path.join(out, asset))
                files.append((asset, entry, os.path.join(out, asset), True))
            else:
                files.append((f"{name}--{entry}", entry, p, False))
        listed = []
        for asset, rel, src, unzip in files:
            dest = os.path.join(out, asset)
            if src != dest:
                shutil.copyfile(src, dest)
            listed.append(
                {"asset": asset, "path": rel, "size": os.path.getsize(dest), "sha256": sha256(dest), "unzip": unzip}
            )
        manifest["packs"][name] = {
            "title": pack["title"],
            "license": pack["license"],
            "source": BASE + pack["archive"],
            "bytes": sum(f["size"] for f in listed),
            "files": listed,
        }
    if gallery:
        with open(os.path.join(gallery, "gallery.json")) as f:
            voices = json.load(f)["voices"]
        for v in voices:
            shutil.copyfile(os.path.join(gallery, v["asset"]), os.path.join(out, v["asset"]))
        shutil.copyfile(os.path.join(gallery, "gallery.json"), os.path.join(out, "gallery.json"))
        manifest["gallery"] = {
            "asset": "gallery.json",
            "voices": len(voices),
            "sha256": sha256(os.path.join(out, "gallery.json")),
        }
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps(manifest, indent=1))


def publish(out):
    env = dict(os.environ, GH_PAGER="cat")
    exists = subprocess.run(["gh", "release", "view", RELEASE], env=env, capture_output=True).returncode == 0
    if not exists:
        subprocess.run(
            [
                "gh", "release", "create", RELEASE, "--prerelease",
                "--title", "Narrator Studio voice models",
                "--notes", "Free voice models republished unpacked for the Narrator Studio app "
                "(downloaded from inside the app). Sources and licenses are in manifest.json.",
            ],
            env=env, check=True,
        )
    assets = sorted(os.path.join(out, f) for f in os.listdir(out))
    for i in range(0, len(assets), 20):
        subprocess.run(["gh", "release", "upload", RELEASE, "--clobber", *assets[i : i + 20]], env=env, check=True)
    print(f"uploaded {len(assets)} files to {RELEASE}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fetch", "stage", "publish"])
    ap.add_argument("--work")
    ap.add_argument("--out")
    ap.add_argument("--gallery")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch(a.work)
    elif a.cmd == "stage":
        stage(a.work, a.out, a.gallery)
    else:
        publish(a.out)


if __name__ == "__main__":
    main()
