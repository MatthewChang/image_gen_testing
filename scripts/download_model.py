"""Fetch the Qwen-Image-Edit-2511 snapshot (~57GB bf16) into $HF_HOME."""

import os

from huggingface_hub import snapshot_download

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

path = snapshot_download(
    MODEL_ID,
    max_workers=16,
    # The .jinja chat templates are needed; nothing to exclude, all files are used.
)
print(f"\nsnapshot: {path}")
total = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(path)
    for f in files
    if not os.path.islink(os.path.join(root, f))
)
print(f"on-disk (excluding symlinks): {total / 1e9:.1f} GB")
