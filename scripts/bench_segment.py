"""Text-prompted SAM 3 masks for the robot-swap benchmark.

Run in the sam3_server pixi env (it has the sam3 package + local checkpoint):

    cd ../sam3_server && CUDA_VISIBLE_DEVICES=1 pixi run python \
        ../image_gen/scripts/bench_segment.py

For every source crop in outputs/bench_robot_swap/_inputs/ and every model
output, writes two binary masks next to a masks/ mirror of the tree:
    <stem>__hand.png   the hand (human hand on inputs, robot hand on outputs)
    <stem>__limb.png   hand ∪ arm/forearm
Outputs are resized to the source crop size before segmenting, so masks are
directly comparable. Idempotent: existing masks are skipped.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "outputs/bench_robot_swap"
MASKS = BENCH / "_masks"
SAM3_SERVER = Path(os.environ.get("SAM3_SERVER", ROOT.parent / "sam3_server"))
CKPT = SAM3_SERVER / "checkpoints/sam3.pt"

# Limb = union of every detection across synonyms (robust). Hand = the single
# top-scoring detection of the first phrase that finds anything: a low-score
# second "hand" instance often covers the whole arm, and "robotic hand"
# usually returns the whole robot, so plain "hand" leads on outputs too.
PROMPTS = {
    "input": {"hand": ["hand", "right hand"], "arm": ["arm", "forearm"]},
    "output": {"hand": ["hand", "robot hand", "robotic hand"], "arm": ["robot arm", "robotic arm", "arm"]},
}


def union_mask(proc, state, phrases, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    for p in phrases:
        st = proc.set_text_prompt(p, state)
        if st["masks"].numel():
            m |= st["masks"].any(0)[0].cpu().numpy()
    return m


def top_mask(proc, state, phrases, shape) -> np.ndarray:
    for p in phrases:
        st = proc.set_text_prompt(p, state)
        if st["masks"].numel():
            return st["masks"][int(st["scores"].argmax())][0].cpu().numpy()
    return np.zeros(shape, bool)


def main() -> None:
    sys.path.insert(0, str(SAM3_SERVER / "third_party/sam3"))
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    model = build_sam3_image_model(checkpoint_path=str(CKPT), load_from_HF=False)
    proc = Sam3Processor(model, confidence_threshold=0.4)

    sources = {p.stem: p for p in sorted((BENCH / "_inputs").glob("*.png"))}
    todo = [("input", p, sources[p.stem]) for p in sources.values()]
    for model_dir in sorted(d for d in BENCH.iterdir() if d.is_dir() and not d.name.startswith("_")):
        for p in sorted(model_dir.glob("*.png")):
            todo.append(("output", p, sources[p.stem.split("__")[0]]))

    done = 0
    for kind, path, src in todo:
        rel = path.relative_to(BENCH).with_suffix("")
        hand_p, limb_p = MASKS / f"{rel}__hand.png", MASKS / f"{rel}__limb.png"
        if hand_p.exists() and limb_p.exists():
            continue
        hand_p.parent.mkdir(parents=True, exist_ok=True)
        size = Image.open(src).size
        img = Image.open(path).convert("RGB").resize(size, Image.LANCZOS)
        shape = (size[1], size[0])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            state = proc.set_image(img)
            hand = top_mask(proc, state, PROMPTS[kind]["hand"], shape)
            limb = hand | union_mask(proc, state, PROMPTS[kind]["hand"] + PROMPTS[kind]["arm"], shape)
        Image.fromarray(hand.astype(np.uint8) * 255).save(hand_p)
        Image.fromarray(limb.astype(np.uint8) * 255).save(limb_p)
        done += 1
        print(f"{rel}: hand {hand.mean():.3f} limb {limb.mean():.3f}", flush=True)
    print(f"segmented {done} images ({len(todo) - done} cached)")


if __name__ == "__main__":
    main()
