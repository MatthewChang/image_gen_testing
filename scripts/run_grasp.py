"""Egocentric images of an object held in a *functional* grasp, on inputs/counter.jpeg.

The inverse of run_counter.py: there the hand vocabulary was pushed into the
negative prompt, here we want exactly one hand and one forearm and nothing else
of the person. That "nothing else" is the hard part — asking for a hand tends to
drag in the second hand, a torso, or a face — so the exclusions still live in the
negative prompt and the positive prompt spends its words on the grasp itself.

A "functional grasp" is object-specific: the grip a person actually uses to do
the thing the object is for, so each object carries its own grasp description in
GRASPS rather than a generic "holding it".

See GRASP_GUIDE.md to add an object, or to diagnose a render that came out wrong.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import DEFAULTS, crop_to_ratio, dims_for_ratio, load, parse_ratio

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT / "inputs" / "counter.jpeg"

# One hand, one forearm, nothing else. "two hands" and "both hands" are the most
# common failures; the rest of the body follows if a shoulder gets a foothold.
NEGATIVE = (
    "two hands, both hands, second hand, left hand, extra hand, extra arm, extra fingers, "
    "six fingers, missing fingers, deformed hand, mangled fingers, fused fingers, "
    "torso, chest, shoulder, upper arm, face, head, hair, legs, feet, full body, "
    "another person, second person, third-person view, selfie, mirror, reflection of a person, "
    "glove, gloves, long sleeve covering the hand, "
    "floating disembodied hand, hand detached from arm, blurry, deformed, low quality"
)

# (slug, object description, functional grasp description)
#
# These four span different grip types *and* different resting poses, which is
# what stresses the object-agnostic placement prompt in run_place_down.py.
GRASPS = {
    "pizza-peel": (
        "a large flat wooden pizza peel with a long straight handle",
        "The hand grips the very end of the long wooden handle in a firm closed power grip, "
        "all four fingers wrapped right around the underside of the handle and the thumb "
        "closed over the top of it, knuckles visible from above. The handle runs away from "
        "the hand toward the upper part of the frame, and the wide flat blade of the peel is "
        "held level in mid-air well clear of the countertop, as if about to slide a pizza "
        "into an oven"
    ),
    "knife": (
        "a kitchen knife with a wide steel blade and a dark moulded handle",
        "The hand holds the knife in a firm power grip with all four fingers wrapped right "
        "around the handle and the thumb pressed flat along the side of the handle just behind "
        "the bolster, the index finger closest to the blade. The blade points away from the "
        "hand toward the upper left of the frame, angled slightly down toward the counter and "
        "held a short distance above it, as if about to start slicing"
    ),
    "mug": (
        "a plain ceramic mug with a curved side handle",
        "The index and middle fingers are hooked through the mug's handle in a hook grip with "
        "the thumb resting on top of the handle, the remaining fingers curled loosely beneath "
        "it. The mug hangs upright and level from the handle, its open top facing the ceiling, "
        "lifted clear of the counter as if being raised to drink from"
    ),
    "mouse": (
        "a wireless computer mouse with a smooth curved plastic shell and a scroll wheel",
        "The palm is draped over the curved top shell of the mouse in a relaxed palmar grip, "
        "the index finger resting on the left button and the middle finger on the right button "
        "with the scroll wheel between them, the thumb along the near side of the shell and the "
        "remaining two fingers along the far side. The mouse is lifted a short way off the "
        "counter and held level, still the right way up"
    ),
}

# Three phrasings, from most anatomically explicit to most terse. Hands are where
# diffusion models fail, and prompt length cuts both ways — more grasp detail can
# fix the grip or can over-constrain and warp the fingers — so the terse variant
# is a genuine control, not filler.
VARIANTS = {
    # Full spatial spec: where the arm enters frame, where the object sits, what's excluded.
    "explicit": (
        "A first person point-of-view photo taken from a head-mounted camera, looking down "
        "and slightly forward at a kitchen counter. An adult man's bare right hand and "
        "forearm reach in from the bottom right of the frame, entering from just below the "
        "camera, holding {obj}. {grasp}. The forearm is bare to just below the elbow, with "
        "visible skin texture and forearm hair, and it is cut off by the bottom edge of the "
        "frame — the arm is the only part of the person in the picture. Natural indoor "
        "kitchen lighting, sharp focus on the hand and the object, realistic photo."
    ),
    # Natural photographic phrasing — leans on "GoPro chest-height POV" as a learned framing prior.
    "action": (
        "GoPro head-mounted camera footage, first person view looking down at a kitchen "
        "counter. I am holding {obj} in my right hand, lifted up off the counter in front of "
        "me. {grasp}. My bare forearm comes up into the bottom right of the frame and is "
        "cropped by the frame edge; nothing else of me is visible. Candid realistic photo, "
        "natural kitchen light."
    ),
    # Minimal: grasp plus framing only, no anatomy inventory.
    "terse": (
        "First person head-mounted camera view looking down at a kitchen counter. A single "
        "bare right hand and forearm hold {obj} up off the counter. {grasp}. Only the hand "
        "and forearm are visible, entering from the bottom of the frame. Realistic photo."
    ),
}

SEEDS = [0, 1, 2]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--object", dest="obj_name", default="pizza-peel", choices=sorted(GRASPS))
    ap.add_argument("--ratio", default="16:9", help='output aspect ratio (default 16:9)')
    ap.add_argument("--no-crop", action="store_true", help="skip the pre-crop when using --ratio")
    ap.add_argument("--anchor", default="top", choices=["top", "center"], help="pre-crop anchor")
    ap.add_argument("--out", help="output subdirectory under outputs/ (default: grasp_<object>)")
    args = ap.parse_args()

    obj, grasp = GRASPS[args.obj_name]
    out_dir = ROOT / "outputs" / (args.out or f"grasp_{args.obj_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT).convert("RGB")

    size = {}
    if args.ratio:
        ratio = parse_ratio(args.ratio)
        width, height = dims_for_ratio(ratio)
        size = {"width": width, "height": height}
        if not args.no_crop:
            image = crop_to_ratio(image, ratio, anchor=args.anchor)
        crop = "no crop" if args.no_crop else f"{args.anchor}-crop"
        print(f"output {width}x{height} ({args.ratio}), {crop}", flush=True)

    t0 = time.time()
    pipe = load(quiet=True)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = [(vname, tmpl, seed) for vname, tmpl in VARIANTS.items() for seed in SEEDS]

    results = []
    for i, (vname, tmpl, seed) in enumerate(jobs, 1):
        prompt = tmpl.format(obj=obj, grasp=grasp)
        name = f"{args.obj_name}--{vname}--seed{seed}"
        print(f"[{i}/{len(jobs)}] {name} ...", end="", flush=True)

        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                image=[image],
                prompt=prompt,
                negative_prompt=NEGATIVE,
                true_cfg_scale=DEFAULTS["true_cfg_scale"],
                num_inference_steps=DEFAULTS["num_inference_steps"],
                guidance_scale=DEFAULTS["guidance_scale"],
                num_images_per_prompt=1,
                generator=torch.manual_seed(seed),
                **size,
            )
        elapsed = time.time() - t0

        dest = out_dir / f"{name}.png"
        result = out.images[0]
        result.save(dest)
        print(f" {elapsed:.1f}s", flush=True)
        results.append(
            {"object": args.obj_name, "variant": vname, "seed": seed, "prompt": prompt,
             "output": dest.name, "seconds": round(elapsed, 1), "size": list(result.size)}
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_review(results, obj, grasp, out_dir)
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} images in {total / 60:.1f}min. Review: {out_dir / 'review.html'}")


def write_review(results: list[dict], obj: str, grasp: str, out_dir: Path) -> None:
    """One row per prompt variant, seeds across — so a bad grip reads as
    variant-wide vs. just an unlucky seed."""
    sections = []
    for vname in VARIANTS:
        cells = "".join(
            f"""<figure><img src="{html.escape(r['output'])}">
            <figcaption>seed {r['seed']} &middot; {r['seconds']}s</figcaption></figure>"""
            for r in results
            if r["variant"] == vname
        )
        sections.append(f"""
  <section>
    <h2>{html.escape(vname)}</h2>
    <p class="prompt">{html.escape(VARIANTS[vname].format(obj=obj, grasp=grasp))}</p>
    <div class="grid">{cells}</div>
  </section>""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>functional grasp &mdash; {html.escape(obj)}</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font: 15px/1.55 system-ui, sans-serif; margin: 0 auto; padding: 2rem; max-width: 1500px;
         background: #14161a; color: #e8e8ea; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .meta {{ color: #9aa0a8; margin: 0 0 1rem; font-size: .875rem; }}
  .neg {{ padding: .6rem .8rem; background: #2a1c1c; border-left: 3px solid #a05050; border-radius: 3px;
          font-size: .8125rem; margin: 0 0 1rem; }}
  .grasp {{ padding: .6rem .8rem; background: #1c2a1e; border-left: 3px solid #5a9a5a; border-radius: 3px;
            font-size: .8125rem; margin: 0 0 2rem; }}
  section {{ border-top: 1px solid #2a2e35; padding: 1.5rem 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .5rem; font-family: ui-monospace, monospace; color: #8fd3ff; }}
  .prompt {{ margin: 0 0 1rem; padding: .5rem .7rem; background: #1c1f25; border-left: 3px solid #4a5058;
             border-radius: 3px; font-size: .8125rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }}
  figure {{ margin: 0; }}
  img {{ width: 100%; height: auto; border-radius: 4px; display: block; background: #000; }}
  figcaption {{ font-size: .75rem; color: #9aa0a8; margin-top: .3rem; font-family: ui-monospace, monospace; }}
  .input {{ max-width: 420px; }}
</style>
<h1>Functional grasp &mdash; {html.escape(obj)}</h1>
<p class="meta">{len(results)} images &middot; {len(VARIANTS)} prompt variants &times; {len(SEEDS)} seeds
 &middot; {results[0]['size'][0]}&times;{results[0]['size'][1]} &middot; 40 steps, true_cfg_scale 4.0, bf16</p>
<p class="grasp"><b>grasp spec:</b> {html.escape(grasp)}</p>
<p class="neg"><b>negative_prompt:</b> {html.escape(NEGATIVE)}</p>
<figure class="input"><img src="../../inputs/counter.jpeg"><figcaption>input (backdrop)</figcaption></figure>
{''.join(sections)}
"""
    (out_dir / "review.html").write_text(doc)


if __name__ == "__main__":
    main()
