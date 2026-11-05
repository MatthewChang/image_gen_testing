"""Fetch the robot-swap benchmark checkpoints into ./.hf (HF_HOME).

    pixi run python scripts/bench_download.py [--only firered qwen21 ...]

The distilled `flux-3-image-klein` repo is gated for our account, so the
undistilled `-base` checkpoint (50 steps, guidance 3.5) stands in for it.
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import snapshot_download

REPOS = {
    "firered": ("FireRedTeam/FireRed-Image-Edit-1.1", None),
    "qwen21": ("Qwen/Qwen-Image-2.1-Turbo", None),
    "flux3": (
        "black-forest-labs/flux-3-image-klein-base",
        # Root BF16 package + diffusers-format folders; skip quantized variants.
        ["*.json", "*.md", "model.safetensors", "ae.safetensors", "text_encoder/*",
         "transformer/*", "vae/*"],
    ),
    "flux3pup": ("black-forest-labs/flux-3-image-klein-pup", None),
    "flux2k4b": ("black-forest-labs/FLUX.2-klein-4B", None),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(REPOS))
    args = ap.parse_args()
    for key in args.only or REPOS:
        repo, patterns = REPOS[key]
        print(f"[{key}] {repo} ...", flush=True)
        path = snapshot_download(repo, allow_patterns=patterns, token=os.environ.get("HF_TOKEN"),
                                 max_workers=16)
        print(f"[{key}] done -> {path}", flush=True)


if __name__ == "__main__":
    main()
