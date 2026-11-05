"""Robot-swap benchmark: replace the human right hand/arm in egocentric frames
with a Sharpa Wave hand on a Tianji Marvin arm, across several edit models.

    pixi run -e bench python scripts/bench_robot_swap.py --model qwen2511 --gpu 0
    pixi run -e bench python scripts/bench_robot_swap.py --model flux3 --gpu 3 --seeds 1

Grid per model: every frame in inputs/robot_swap/ego/ x {text, refs} x seeds.
Outputs land in outputs/bench_robot_swap/<model>/<frame>__<cond>__s<seed>.png
with a sidecar .json (prompt actually sent, dims, seconds). Reruns skip
existing outputs (--force regenerates), so a crashed run resumes.

Frames are centre-cropped to exact 16:9 first and the crop is saved to
outputs/bench_robot_swap/_inputs/ — that crop is what every model sees and
what the scorer compares against.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import crop_to_ratio

ROOT = Path(__file__).resolve().parent.parent
EGO = ROOT / "inputs/robot_swap/ego"
REFS = ROOT / "inputs/robot_swap/refs"
OUT = ROOT / "outputs/bench_robot_swap"
RATIO = 16 / 9

PROMPTS = {
    # Text only: the model must know (or guess) what the robot looks like.
    "text": (
        "Replace the person's right hand and forearm with a Sharpa Wave robotic hand mounted on a "
        "Tianji Marvin robot arm. Keep the hand in exactly the same position, finger pose and grip "
        "on the object. Keep everything else unchanged."
    ),
    # With references: image 2 = hand, image 3 = arm.
    "refs": (
        "Replace the person's right hand and forearm in the first image with a Sharpa Wave robotic "
        "hand mounted on a Tianji Marvin robot arm, matching the robotic hand in the second image and "
        "the white robot arm in the third image. Keep the hand in exactly the same position, finger "
        "pose and grip on the object. Keep everything else unchanged."
    ),
}


# ---------------------------------------------------------------------------
# Adapters: load(device) once, then edit(images, prompt, seed) -> PIL image.
# Each uses its model card's recommended sampling settings.
# ---------------------------------------------------------------------------


class QwenEditPlus:
    """Qwen-Image-Edit-2511 and FireRed-Image-Edit-1.1 (same pipeline class)."""

    size = (1280, 720)

    def __init__(self, repo: str):
        self.repo = repo

    def load(self, device: str, offload: bool) -> None:
        from diffusers import QwenImageEditPlusPipeline

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device) if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, images, prompt, seed):
        w, h = self.size
        out = self.pipe(image=images, prompt=prompt, negative_prompt=" ", true_cfg_scale=4.0,
                        guidance_scale=1.0, num_inference_steps=40, width=w, height=h,
                        generator=torch.manual_seed(seed))
        return out.images[0], {"steps": 40, "true_cfg_scale": 4.0}


class Qwen21Turbo:
    """Qwen-Image-2.1-Turbo: 8-step schedule baked into the checkpoint, CFG 1."""

    repo = "Qwen/Qwen-Image-2.1-Turbo"
    size = (2752, 1536)  # the card's 16:9 preset

    def load(self, device: str, offload: bool) -> None:
        from diffusers import QwenImage21Pipeline

        self.pipe = QwenImage21Pipeline.from_pretrained(self.repo, dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device) if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, images, prompt, seed):
        w, h = self.size
        imgs = [im.convert("RGBA") for im in images]
        out = self.pipe(prompt=prompt, image=imgs if len(imgs) > 1 else imgs[0], width=w, height=h,
                        use_kv_cache=True, generator=torch.Generator("cpu").manual_seed(seed))
        return out.images[0], {"steps": 8, "schedule": "checkpoint"}


class Flux2Klein4B:
    """FLUX.2 [klein] 4B distilled: 4 steps, guidance 1.0 (card defaults)."""

    repo = "black-forest-labs/FLUX.2-klein-4B"
    size = (1280, 720)

    def load(self, device: str, offload: bool) -> None:
        from diffusers import Flux2KleinPipeline

        self.device = device
        self.pipe = Flux2KleinPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)
        self.pipe.enable_model_cpu_offload(device=device) if offload else self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, images, prompt, seed):
        w, h = self.size
        out = self.pipe(prompt=prompt, image=images, width=w, height=h, guidance_scale=1.0,
                        num_inference_steps=4, generator=torch.Generator(self.device).manual_seed(seed))
        return out.images[0], {"steps": 4, "guidance_scale": 1.0}


class Flux3KleinBase:
    """FLUX 3 Image Klein (base, undistilled) + its 9B prompt upsampler.

    The distilled checkpoint is gated for our account; base runs at the card's
    50 steps / guidance 3.5 (pipeline defaults). The card says the model must
    not be used without the upsampler, so every request goes through it and
    the upsampled caption is recorded in the sidecar.
    """

    repo = "black-forest-labs/flux-3-image-klein-base"
    pup = "black-forest-labs/flux-3-image-klein-pup"
    size = (1280, 720)

    def load(self, device: str, offload: bool) -> None:
        from diffusers import ModularPipeline
        from huggingface_hub import hf_hub_download
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.device = device
        self.pipe = ModularPipeline.from_pretrained(self.repo)
        self.pipe.load_components(dtype=torch.bfloat16)
        self.pipe.to(device)
        self.processor = AutoProcessor.from_pretrained(self.pup)
        self.upsampler = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.pup, dtype=torch.bfloat16, device_map={"": device}).eval()
        self.system_prompts = json.load(open(hf_hub_download(self.pup, "system_prompts.json")))
        self._captions: dict[tuple, str] = {}

    def upsample(self, prompt, images):
        # Greedy decoding -> deterministic; cache per (prompt, #images, input id).
        key = (prompt, tuple(id(im) for im in images))
        if key in self._captions:
            return self._captions[key]
        text = ("Reference images follow in order (the first image, the second image, ...).\n"
                f"User request: {prompt}\nOutput the caption.")
        content = [{"type": "image", "image": im} for im in images] + [{"type": "text", "text": text}]
        messages = [{"role": "system", "content": self.system_prompts["i2i"]},
                    {"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
            enable_thinking=False).to(self.upsampler.device)
        with torch.no_grad():
            out = self.upsampler.generate(**inputs, max_new_tokens=3584, do_sample=False)
        caption = self.processor.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        self._captions[key] = caption
        return caption

    def edit(self, images, prompt, seed):
        caption = self.upsample(prompt, images)
        w, h = self.size
        img = self.pipe(prompt=caption, image=images if len(images) > 1 else images[0], width=w, height=h,
                        generator=torch.Generator().manual_seed(seed), output="images")[0]
        return img, {"steps": "pipeline default (50)", "guidance": "pipeline default (3.5)",
                     "upsampled_prompt": caption}


MODELS = {
    "qwen2511": lambda: QwenEditPlus("Qwen/Qwen-Image-Edit-2511"),
    "firered11": lambda: QwenEditPlus("FireRedTeam/FireRed-Image-Edit-1.1"),
    "qwen21turbo": Qwen21Turbo,
    "flux3klein": Flux3KleinBase,
    "flux2klein4b": Flux2Klein4B,
}


def prepared_inputs() -> dict[str, Image.Image]:
    """16:9 centre crops of every ego frame, cached under _inputs/."""
    dest = OUT / "_inputs"
    dest.mkdir(parents=True, exist_ok=True)
    frames = {}
    for p in sorted(EGO.glob("*.png")):
        crop = dest / p.name
        if not crop.exists():
            crop_to_ratio(Image.open(p).convert("RGB"), RATIO, anchor="center").save(crop)
        frames[p.stem] = Image.open(crop).convert("RGB")
    return frames


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--offload", action="store_true", help="enable_model_cpu_offload (tight VRAM)")
    ap.add_argument("--seeds", type=int, default=3, help="seeds 0..N-1")
    ap.add_argument("--frames", nargs="*", help="subset of frame stems")
    ap.add_argument("--conds", nargs="*", default=list(PROMPTS), choices=list(PROMPTS))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    frames = prepared_inputs()
    if args.frames:
        frames = {k: v for k, v in frames.items() if k in args.frames}
    refs = [Image.open(REFS / "sharpa_hand.png").convert("RGB"),
            Image.open(REFS / "tianji_arm.png").convert("RGB")]

    out_dir = OUT / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(f, c, s) for f in frames for c in args.conds for s in range(args.seeds)
            if args.force or not (out_dir / f"{f}__{c}__s{s}.png").exists()]
    print(f"[{args.model}] {len(jobs)} edits to run", flush=True)
    if not jobs:
        return

    device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    model = MODELS[args.model]()
    t0 = time.time()
    model.load(device, args.offload)
    print(f"[{args.model}] loaded in {time.time() - t0:.0f}s on {device}"
          f"{' (offload)' if args.offload else ''}", flush=True)

    for i, (frame, cond, seed) in enumerate(jobs, 1):
        images = [frames[frame]] + (refs if cond == "refs" else [])
        t = time.time()
        with torch.inference_mode():
            img, meta = model.edit(images, PROMPTS[cond], seed)
        secs = time.time() - t
        stem = out_dir / f"{frame}__{cond}__s{seed}"
        img.convert("RGB").save(stem.with_suffix(".png"))
        stem.with_suffix(".json").write_text(json.dumps(
            {"model": args.model, "frame": frame, "cond": cond, "seed": seed, "prompt": PROMPTS[cond],
             "size": list(img.size), "seconds": round(secs, 1), **meta}, indent=2))
        print(f"[{args.model}] {i}/{len(jobs)} {stem.name} {img.size[0]}x{img.size[1]} {secs:.1f}s",
              flush=True)


if __name__ == "__main__":
    main()
