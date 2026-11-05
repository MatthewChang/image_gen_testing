#!/usr/bin/env python
"""Client for scripts/qwen_server.py. Same flags as scripts/edit.py, plus --url.

    pixi run -e bench client -i inputs/cat.png -p "add a red bow tie" -o outputs/srv/bow.png \
        [--url http://127.0.0.1:8100] [--seed 0] [--steps 40] [--cfg 4.0] [--ratio 16:9]

From Python (paths must be visible to the server, i.e. on shared NFS):

    from qwen_client import edit
    res = edit("http://127.0.0.1:8100", ["inputs/cat.png"], "add a red bow tie", "outputs/srv/bow.png", seed=1)
"""

from __future__ import annotations

import argparse
import json
import os
import time

import requests


def wait_ready(url: str, timeout: float = 600) -> dict:
    """Poll /health until the model is loaded; return the health dict."""
    t0 = time.time()
    while True:
        try:
            h = requests.get(f"{url}/health", timeout=5).json()
            if h["status"] == "ready":
                return h
        except requests.ConnectionError:
            pass
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{url} not ready after {timeout:.0f}s")
        time.sleep(2)


def edit(url: str, images: list[str], prompt: str, output: str, wait: float = 600, **kw) -> dict:
    """Send one edit. kw: seed, steps, true_cfg_scale, negative_prompt, ratio, anchor, width, height, name.

    Retries while the server is busy (503), up to `wait` seconds.
    """
    payload = {"images": [os.path.abspath(p) for p in images], "prompt": prompt,
               "output_path": os.path.abspath(output), **{k: v for k, v in kw.items() if v is not None}}
    payload.setdefault("name", os.path.basename(output))
    t0 = time.time()
    while True:
        r = requests.post(f"{url}/process", json=payload)
        if r.status_code != 503 or time.time() - t0 > wait:
            break
        time.sleep(2)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("-i", "--image", action="append", required=True, help="input image (repeat for multi-image)")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--steps", type=int, default=None, help="default: model default")
    ap.add_argument("--cfg", type=float, default=None, help="true_cfg_scale (default: model default)")
    ap.add_argument("--negative", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ratio", help='output aspect ratio, e.g. "16:9"')
    ap.add_argument("--anchor", default=None, choices=["top", "center", "none"], help="pre-crop anchor for --ratio")
    ap.add_argument("--size", help="explicit WxH, e.g. 1280x720")
    args = ap.parse_args()

    width = height = None
    if args.size:
        width, height = (int(v) for v in args.size.lower().split("x"))
    h = wait_ready(args.url)
    print(f"server ready: {h['meta']}", flush=True)
    res = edit(args.url, args.image, args.prompt, args.output, seed=args.seed, steps=args.steps,
               true_cfg_scale=args.cfg, negative_prompt=args.negative, ratio=args.ratio, anchor=args.anchor,
               width=width, height=height)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
