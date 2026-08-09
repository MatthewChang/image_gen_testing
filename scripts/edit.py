"""Edit one or more images with Qwen-Image-Edit-2511.

    pixi run edit -i cat.png -p "make it wear a wizard hat" -o out.png
    pixi run edit -i bear1.png -i bear2.png -p "the two bears shake hands" -o out.png
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import DEFAULTS, crop_to_ratio, dims_for_ratio, load, parse_ratio


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--image", action="append", required=True, help="input image (repeat for multi-image edits)")
    ap.add_argument("-p", "--prompt", required=True)
    ap.add_argument("-o", "--output", required=True)
    ap.add_argument("-n", "--steps", type=int, default=DEFAULTS["num_inference_steps"])
    ap.add_argument("--cfg", type=float, default=DEFAULTS["true_cfg_scale"], help="true_cfg_scale")
    ap.add_argument("--negative", default=DEFAULTS["negative_prompt"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ratio",
        help='output aspect ratio, e.g. "16:9". Default: inherit the input image\'s ratio.',
    )
    ap.add_argument(
        "--no-crop",
        action="store_true",
        help="with --ratio, reframe freely instead of cropping the input to match first",
    )
    ap.add_argument("--anchor", default="top", choices=["top", "center"], help="pre-crop anchor")
    args = ap.parse_args()

    images = [Image.open(p).convert("RGB") for p in args.image]

    size = {}
    if args.ratio:
        ratio = parse_ratio(args.ratio)
        width, height = dims_for_ratio(ratio)
        size = {"width": width, "height": height}
        if not args.no_crop:
            images = [crop_to_ratio(im, ratio, anchor=args.anchor) for im in images]

    pipe = load(quiet=False)

    t0 = time.time()
    with torch.inference_mode():
        out = pipe(
            image=images,
            prompt=args.prompt,
            negative_prompt=args.negative,
            true_cfg_scale=args.cfg,
            num_inference_steps=args.steps,
            guidance_scale=DEFAULTS["guidance_scale"],
            num_images_per_prompt=1,
            generator=torch.manual_seed(args.seed),
            **size,
        )

    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.images[0].save(dest)
    print(f"{dest}  ({out.images[0].size[0]}x{out.images[0].size[1]}, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
