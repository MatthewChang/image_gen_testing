""" objects on inputs/counter.jpeg from a first-person head-camera viewpoint.

The hand exclusion lives in the *negative* prompt, not the positive one — naming
"hands" positively (even as "no hands visible") reliably summons a hand, since
diffusion conditioning has no negation operator. The positive prompt likewise
describes the object's *position* rather than the act of grasping it, so the
model has no reason to depict a grasp.

Sweeps two prompt phrasings x two seeds per object so the variants are
comparable side by side.
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

# Everything hand/person-shaped goes here instead of into the prompt.
NEGATIVE = (
    "hand, hands, fingers, thumb, fingertip, arm, arms, wrist, forearm, elbow, "
    "person, people, human, body, holding, grasping, gripping, reaching, "
    "pointing, touching the object, someone holding it"
)

# Two positive phrasings. Neither mentions hands, grasping, or the wearer's body.
# "reach" describes placement via distance; "plain" omits reachability entirely
# and relies on composition alone — the control for whether the clause hurts.
VARIANTS = {
    "reach": (
        "a first person point-of-view photo from a head-mounted camera, looking down at a "
        "kitchen counter. {obj} sits alone on the empty countertop, slightly right of centre "
        "and close to the near edge, within arm's reach of the camera. The object rests "
        "unobstructed and fully visible, casting a soft natural shadow on the counter. "
        "Nothing else is on the counter. Empty room, nobody present."
    ),
    "plain": (
        "a first person point-of-view photo from a head-mounted camera, looking down at a "
        "kitchen counter. {obj} sits alone on the empty countertop, unobstructed and fully "
        "visible, casting a soft natural shadow. Nothing and nobody else is in the frame."
    ),
}

OBJECT_SETS = {
    # (slug, object description) — the description fills {obj} in each variant.
    "household": [
        ("cup", "a ceramic cup"),
        ("pizza-peel", "a large flat wooden pizza peel with a long straight handle"),
        (
            "workbook",
            "a thin floppy paperback workbook lying flat and closed, only a few millimetres "
            "thick, with a soft bendy paper cover and no spine",
        ),
        (
            "violin",
            "a Stradivarius violin lying flat on its back, with an hourglass-shaped varnished "
            "wooden body and a slender scrolled neck",
        ),
    ],
    "tools": [
        ("mouse", "a computer mouse, wireless, with a smooth curved plastic shell and a scroll wheel"),
        (
            "controller",
            "a video game controller with two thumbsticks, a directional pad and face buttons, "
            "lying flat with its grips pointing toward the camera",
        ),
        (
            "knife",
            "a kitchen knife lying flat with its blade and handle in a straight line, "
            "a wide steel blade and a dark moulded handle",
        ),
        (
            "l-bracket",
            "a small metal L-angle bracket, a right-angled piece of galvanised steel with "
            "screw holes drilled through both flanges",
        ),
    ],
}

SEEDS = [0, 1]

# Where each set's outputs land, relative to outputs/.
OUT_DIRS = {"household": "counter", "tools": "counter_tools"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--set", dest="set_name", default="household", choices=sorted(OBJECT_SETS))
    ap.add_argument("--ratio", help='output aspect ratio, e.g. "16:9" (default: the input\'s)')
    ap.add_argument("--no-crop", action="store_true", help="skip the pre-crop when using --ratio")
    ap.add_argument("--anchor", default="top", choices=["top", "center"], help="pre-crop anchor")
    ap.add_argument("--out", help="output subdirectory under outputs/ (default: per --set)")
    args = ap.parse_args()

    objects = OBJECT_SETS[args.set_name]
    out_dir = ROOT / "outputs" / (args.out or OUT_DIRS[args.set_name])
    out_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(INPUT).convert("RGB")

    size = {}
    if args.ratio:
        ratio = parse_ratio(args.ratio)
        width, height = dims_for_ratio(ratio)
        size = {"width": width, "height": height}
        if not args.no_crop:
            image = crop_to_ratio(image, ratio, anchor=args.anchor)
        crop = f"{args.anchor}-crop" if not args.no_crop else "no crop"
        print(f"output {width}x{height} ({args.ratio}), {crop}", flush=True)

    t0 = time.time()
    pipe = load(quiet=True)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = [
        (slug, obj, vname, tmpl, seed)
        for slug, obj in objects
        for vname, tmpl in VARIANTS.items()
        for seed in SEEDS
    ]

    results = []
    for i, (slug, obj, vname, tmpl, seed) in enumerate(jobs, 1):
        prompt = tmpl.format(obj=obj)
        name = f"{slug}--{vname}--seed{seed}"
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
            {"slug": slug, "object": obj, "variant": vname, "seed": seed, "prompt": prompt,
             "output": dest.name, "seconds": round(elapsed, 1), "size": list(result.size)}
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_review(results, objects, out_dir)
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} images in {total / 60:.1f}min. Review: {out_dir / 'review.html'}")


def write_review(results: list[dict], objects: list[tuple[str, str]], out_dir: Path) -> None:
    """Group by object so the four variant/seed cells sit in one row per object."""
    sections = []
    for slug, obj in objects:
        cells = "".join(
            f"""<figure><img src="{html.escape(r['output'])}">
            <figcaption>{html.escape(r['variant'])} &middot; seed {r['seed']}</figcaption></figure>"""
            for r in results
            if r["slug"] == slug
        )
        prompts = "".join(
            f'<p class="prompt"><b>{html.escape(v)}</b>: {html.escape(t.format(obj=obj))}</p>'
            for v, t in VARIANTS.items()
        )
        sections.append(f"""
  <section>
    <h2>{html.escape(slug)}</h2>
    {prompts}
    <div class="grid">{cells}</div>
  </section>""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>counter.jpeg &mdash; object placement sweep</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font: 15px/1.55 system-ui, sans-serif; margin: 0 auto; padding: 2rem; max-width: 1400px;
         background: #14161a; color: #e8e8ea; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .meta {{ color: #9aa0a8; margin: 0 0 1rem; font-size: .875rem; }}
  .neg {{ padding: .6rem .8rem; background: #2a1c1c; border-left: 3px solid #a05050; border-radius: 3px;
          font-size: .8125rem; margin: 0 0 2rem; }}
  section {{ border-top: 1px solid #2a2e35; padding: 1.5rem 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .5rem; font-family: ui-monospace, monospace; color: #8fd3ff; }}
  .prompt {{ margin: 0 0 .5rem; padding: .5rem .7rem; background: #1c1f25; border-left: 3px solid #4a5058;
             border-radius: 3px; font-size: .8125rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: .75rem; margin-top: 1rem; }}
  figure {{ margin: 0; }}
  img {{ width: 100%; height: auto; border-radius: 4px; display: block; background: #000; }}
  figcaption {{ font-size: .75rem; color: #9aa0a8; margin-top: .3rem; font-family: ui-monospace, monospace; }}
  .input {{ max-width: 420px; }}
</style>
<h1>counter.jpeg &mdash; first-person object placement (v2)</h1>
<p class="meta">{len(results)} images &middot; 2 prompt variants &times; 2 seeds per object
 &middot; 40 steps, true_cfg_scale 4.0, bf16</p>
<p class="neg"><b>negative_prompt:</b> {html.escape(NEGATIVE)}</p>
<figure class="input"><img src="../../inputs/counter.jpeg"><figcaption>input</figcaption></figure>
{''.join(sections)}
"""
    (out_dir / "review.html").write_text(doc)


if __name__ == "__main__":
    main()
