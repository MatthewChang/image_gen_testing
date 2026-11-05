#!/usr/bin/env python
"""Qwen image-edit server: one edit model resident on one GPU, behind model_serving.

    CUDA_VISIBLE_DEVICES=3 pixi run -e bench serve --model qwen2511 --port 8100
    curl -s localhost:8100/health
    pixi run -e bench client -i inputs/cat.png -p "add a red bow tie" -o outputs/srv/bow.png \
        --url http://127.0.0.1:8100

--model (one model per process; run several processes for several models):
    qwen2511     Qwen/Qwen-Image-Edit-2511           40 steps, true_cfg_scale 4.0
    firered11    FireRedTeam/FireRed-Image-Edit-1.1  same pipeline class and defaults as 2511
    qwen21turbo  Qwen/Qwen-Image-2.1-Turbo           checkpoint sigma schedule (8 steps), cfg 1.0

Request (paths are on the shared filesystem; nothing is sent inline):
    images          input image paths; image 1 is the main one, the rest are references
    prompt          edit instruction
    output_path     PNG written here, plus a .json sidecar with the parameters used
    seed            CPU generator seed (default 0; same stream as scripts/edit.py)
    steps           denoising steps (default: model default; not supported by qwen21turbo)
    true_cfg_scale  default: model default
    negative_prompt default " "
    ratio           e.g. "16:9": output dims from pipeline.dims_for_ratio, inputs pre-cropped
    anchor          pre-crop anchor for ratio: "top" (default) | "center" | "none" (no crop)
    width, height   explicit output dims (override ratio's dims; both or neither)

Default output size is the pipeline's own: 2511/FireRed follow the FIRST image's aspect
ratio at ~1MP; 2.1-Turbo follows the LAST image's aspect ratio at ~1MP.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Literal

import torch
from PIL import Image
from pydantic import BaseModel, Field

from model_serving import ModelServer
from pipeline import crop_to_ratio, dims_for_ratio, parse_ratio


class EditRequest(BaseModel):
    name: str | None = None
    images: list[str] = Field(min_length=1)
    prompt: str
    output_path: str
    seed: int = 0
    steps: int | None = None
    true_cfg_scale: float | None = None
    negative_prompt: str = " "
    ratio: str | None = None
    anchor: Literal["top", "center", "none"] = "top"
    width: int | None = None
    height: int | None = None


class EditResponse(BaseModel):
    output_path: str
    width: int
    height: int
    seconds: float
    model: str
    params: dict


class Cancelled(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Adapters: load(device) once, then edit(images, req, size, generator, callback) -> (PIL, params)
# ---------------------------------------------------------------------------


class QwenEditPlus:
    """Qwen-Image-Edit-2511 and FireRed-Image-Edit-1.1 (QwenImageEditPlusPipeline)."""

    def __init__(self, repo: str):
        self.repo = repo

    def load(self, device: str) -> None:
        from diffusers import QwenImageEditPlusPipeline

        self.pipe = QwenImageEditPlusPipeline.from_pretrained(self.repo, torch_dtype=torch.bfloat16)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, images, req, size, generator, callback):
        params = {"steps": req["steps"] or 40,
                  "true_cfg_scale": 4.0 if req["true_cfg_scale"] is None else req["true_cfg_scale"],
                  "negative_prompt": req["negative_prompt"]}
        out = self.pipe(image=images, prompt=req["prompt"], negative_prompt=params["negative_prompt"],
                        true_cfg_scale=params["true_cfg_scale"], num_inference_steps=params["steps"],
                        guidance_scale=1.0, generator=generator, callback_on_step_end=callback, **size)
        return out.images[0], params


class Qwen21Turbo:
    """Qwen-Image-2.1-Turbo: sigma schedule baked into the checkpoint, CFG 1."""

    repo = "Qwen/Qwen-Image-2.1-Turbo"

    def load(self, device: str) -> None:
        from diffusers import QwenImage21Pipeline

        self.pipe = QwenImage21Pipeline.from_pretrained(self.repo, dtype=torch.bfloat16)
        self.pipe.to(device)
        self.pipe.set_progress_bar_config(disable=True)

    def edit(self, images, req, size, generator, callback):
        if req["steps"] is not None:
            # num_inference_steps is ignored whenever the checkpoint config sets sample_sigmas.
            raise ValueError("qwen21turbo uses the checkpoint's sigma schedule; do not set steps")
        params = {"steps": "checkpoint schedule",
                  "true_cfg_scale": 1.0 if req["true_cfg_scale"] is None else req["true_cfg_scale"],
                  "negative_prompt": req["negative_prompt"]}
        imgs = [im.convert("RGBA") for im in images]
        out = self.pipe(prompt=req["prompt"], image=imgs if len(imgs) > 1 else imgs[0],
                        negative_prompt=params["negative_prompt"], true_cfg_scale=params["true_cfg_scale"],
                        use_kv_cache=True, generator=generator, callback_on_step_end=callback, **size)
        return out.images[0], params


MODELS = {
    "qwen2511": lambda: QwenEditPlus("Qwen/Qwen-Image-Edit-2511"),
    "firered11": lambda: QwenEditPlus("FireRedTeam/FireRed-Image-Edit-1.1"),
    "qwen21turbo": Qwen21Turbo,
}


class QwenEditServer(ModelServer):
    """Qwen image-edit server (see scripts/qwen_server.py docstring for the request fields)."""

    request_model = EditRequest
    response_model = EditResponse
    model_key = "qwen2511"  # set from --model in main()

    def setup(self) -> None:
        self.meta = {"model": self.model_key, "repo": getattr(self.adapter, "repo", None), "task": "image-edit"}
        self.adapter.load("cuda")

    def process(self, req: dict, cancel: threading.Event) -> dict:
        try:
            return self._edit(req, cancel)
        finally:
            torch.cuda.empty_cache()

    def _edit(self, req: dict, cancel: threading.Event) -> dict:
        if (req["width"] is None) != (req["height"] is None):
            raise ValueError("give both width and height, or neither")
        images = [Image.open(p).convert("RGB") for p in req["images"]]
        size = {}
        if req["ratio"]:
            ratio = parse_ratio(req["ratio"])
            w, h = dims_for_ratio(ratio)
            size = {"width": w, "height": h}
            if req["anchor"] != "none":
                images = [crop_to_ratio(im, ratio, anchor=req["anchor"]) for im in images]
        if req["width"] is not None:
            size = {"width": req["width"], "height": req["height"]}

        def on_step(pipe, i, t, kwargs):
            if cancel.is_set():
                raise Cancelled(f"client disconnected at step {i}")
            return {}

        t0 = time.time()
        with torch.inference_mode():
            img, params = self.adapter.edit(images, req, size, torch.Generator("cpu").manual_seed(req["seed"]),
                                            on_step)
        secs = round(time.time() - t0, 1)

        dest = Path(req["output_path"])
        dest.parent.mkdir(parents=True, exist_ok=True)
        img = img.convert("RGB")
        img.save(dest)
        params = {**params, "seed": req["seed"], "ratio": req["ratio"], "anchor": req["anchor"], **size}
        dest.with_suffix(".json").write_text(json.dumps(
            {"model": self.model_key, "images": req["images"], "prompt": req["prompt"],
             "size": list(img.size), "seconds": secs, **params}, indent=2))
        return {"output_path": str(dest), "width": img.size[0], "height": img.size[1], "seconds": secs,
                "model": self.model_key, "params": params}


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", default="qwen2511", choices=list(MODELS))
    args, rest = ap.parse_known_args()
    QwenEditServer.model_key = args.model
    QwenEditServer.adapter = MODELS[args.model]()
    QwenEditServer.serve(argv=rest)


if __name__ == "__main__":
    main()
