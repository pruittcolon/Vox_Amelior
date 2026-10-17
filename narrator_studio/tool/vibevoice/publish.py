#!/usr/bin/env python3
"""Stages the VibeVoice pack for the narrator-models release and merges it
into the release's manifest.json (the app's download catalog).

  publish.py --pack DIR --manifest manifest.json --out STAGE

Every file becomes an asset named vibevoice--<hash>--<path with / as -->; STAGE then
holds the assets and the updated manifest.json, ready for `gh release upload`.
"""

import argparse
import hashlib
import json
import os
import shutil

TITLE = "Lifelike voices (VibeVoice-Realtime-0.5B)"
LICENSE = "MIT (microsoft/VibeVoice-Realtime-0.5B), converted to ONNX by tool/vibevoice"
SOURCE = "https://huggingface.co/microsoft/VibeVoice-Realtime-0.5B"
SKIP = {"shrink.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    files = []
    for root, _, names in os.walk(a.pack):
        for n in sorted(names):
            src = os.path.join(root, n)
            rel = os.path.relpath(src, a.pack).replace(os.sep, "/")
            if rel in SKIP:
                continue
            # The content hash is in the name: a new version never replaces
            # a file an installed app may be downloading.
            digest = sha256(src)
            asset = f"vibevoice--{digest[:10]}--" + rel.replace("/", "--")
            dest = os.path.join(a.out, asset)
            shutil.copyfile(src, dest)
            files.append({"asset": asset, "path": rel, "size": os.path.getsize(dest), "sha256": digest, "unzip": False})
    files.sort(key=lambda f: f["path"])
    manifest = json.load(open(a.manifest))
    manifest["packs"]["vibevoice"] = {
        "title": TITLE,
        "license": LICENSE,
        "source": SOURCE,
        "bytes": sum(f["size"] for f in files),
        "files": files,
    }
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"vibevoice: {len(files)} files, {manifest['packs']['vibevoice']['bytes'] / 1e6:.0f} MB")
    for f in files:
        print(f"  {f['path']}: {f['size'] / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
