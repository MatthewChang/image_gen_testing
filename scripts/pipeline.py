"""Shared loader for the Qwen-Image-Edit-2511 pipeline."""

from __future__ import annotations

import math

import torch
from diffusers import QwenImageEditPlusPipeline
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-Edit-2511"

# The transformer packs latents into 2x2 patches, so both dimensions must be a
# multiple of vae_scale_factor * 2 == 16. The pipeline floors to this silently;
# we round explicitly so the requested ratio is what actually comes out.
MULTIPLE_OF = 16

# Defaults from the model card.
DEFAULTS = {
    "true_cfg_scale": 4.0,
    "negative_prompt": " ",
    "num_inference_steps": 40,
    "guidance_scale": 1.0,
    "num_images_per_prompt": 1,
}


def load(device: str = "cuda", quiet: bool = True) -> QwenImageEditPlusPipeline:
    """Load the 20B edit pipeline in bf16 onto a single device.

    An H200 (143GB) holds the whole thing resident, so no offloading is needed.
    """
    pipe = QwenImageEditPlusPipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True if quiet else None)
    return pipe


def dims_for_ratio(ratio: float, target_area: int = 1024 * 1024) -> tuple[int, int]:
    """(width, height) at `ratio`, near target_area pixels, both multiples of 16.

    Naively rounding sqrt(area * ratio) to 16 leaves the ratio slightly off
    (16:9 lands on 1360x768 = 1.771). Instead we scan candidate heights and take
    the closest ratio match, tie-breaking on area — which finds the exact pairs
    where they exist (16:9 -> 1280x720).
    """
    ideal_h = math.sqrt(target_area / ratio)
    best = None
    for h in range(MULTIPLE_OF, 2049, MULTIPLE_OF):
        # Only consider heights within 25% of ideal, so we stay near target_area.
        if not 0.75 * ideal_h <= h <= 1.25 * ideal_h:
            continue
        w = round(h * ratio / MULTIPLE_OF) * MULTIPLE_OF
        if w < MULTIPLE_OF:
            continue
        score = (abs(w / h - ratio), abs(w * h - target_area))
        if best is None or score < best[0]:
            best = (score, (w, h))
    if best is None:  # extreme ratio - fall back to the naive rounding
        w = round(math.sqrt(target_area * ratio) / MULTIPLE_OF) * MULTIPLE_OF
        return max(w, MULTIPLE_OF), max(round(w / ratio / MULTIPLE_OF) * MULTIPLE_OF, MULTIPLE_OF)
    return best[1]


def parse_ratio(spec: str) -> float:
    """Accept "16:9", "16x9" or a bare float like "1.777"."""
    for sep in (":", "x", "/"):
        if sep in spec:
            w, h = spec.split(sep, 1)
            return float(w) / float(h)
    return float(spec)


def crop_to_ratio(image: Image.Image, ratio: float, anchor: str = "top") -> Image.Image:
    """Crop the largest region matching `ratio`, keeping `anchor` ("top"|"center").

    Worth doing before a ratio change: the condition image is encoded at its own
    aspect ratio, so feeding a 4:3 source while asking for 16:9 output makes the
    model invent the side content it was never shown. Cropping first keeps the
    conditioning and the target frame consistent.

    Anchor matters more than it looks. On a downward-facing shot the background
    that anchors the scene (here: fridge, cabinets, floor) sits in the top
    strip; a centre crop discards it and the model reinvents the room. Default
    to "top" so that context survives.
    """
    w, h = image.size
    if w / h > ratio:  # too wide - trim the sides, always centred
        new_w = round(h * ratio)
        left = (w - new_w) // 2
        return image.crop((left, 0, left + new_w, h))
    new_h = round(w / ratio)  # too tall - trim vertically per anchor
    top = 0 if anchor == "top" else (h - new_h) // 2
    return image.crop((0, top, w, top + new_h))


# Back-compat alias for the original name.
def center_crop_to_ratio(image: Image.Image, ratio: float) -> Image.Image:
    return crop_to_ratio(image, ratio, anchor="center")
