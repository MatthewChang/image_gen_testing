"""Invert a placed-down image: pick the object up off the counter in a functional grasp.

The reverse of run_place_down.py. Input is an on-counter frame (the *_before.png
files under outputs/remove_pairs/<object>/, i.e. the outputs of the placing edit),
and the target is the "after pickup" frame: the same object lifted clear of the
counter, held in the grip a person actually uses to use it, with exactly one bare
hand and forearm in frame.

Where the placing direction had one open question (is "remove the hand" an
instruction or content?), this direction has a different one: does the model need
to be *told* the functional grip, or can it infer it from the object it can see?
run_grasp.py needed hand-written per-object grasp specs because from-scratch
generation had nothing to anchor on — but here the conditioning image already
shows the object, so maybe "the grip a person actually uses for its purpose" is
enough. The two variants split exactly on that:

  add    - object-agnostic, like the placing prompts: describes a functional
           grip abstractly and lets the model derive the finger placement from
           the object in frame
  spec   - same edit instruction, but with the object's hand-written grasp
           description from run_grasp.GRASPS spliced in

If "add" holds up, new objects need no grasp spec for this direction at all.

The negative prompt is run_grasp's (one hand, no second hand, no body) plus
terms for the two failure modes specific to editing: the object staying put on
the counter, and a hand merely hovering near it without contact.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import DEFAULTS, load
from run_grasp import GRASPS, NEGATIVE as GRASP_NEGATIVE
from run_place_down import hstack

ROOT = Path(__file__).resolve().parent.parent
PAIRS_ROOT = ROOT / "outputs" / "remove_pairs"

NEGATIVE = GRASP_NEGATIVE + (
    ", object resting on the counter, object still sitting on the countertop, "
    "object untouched on the surface, hand hovering near the object without touching it, "
    "hand not in contact with the object, empty hand"
)

# The shared skeleton of both variants: what gets added, where the arm enters,
# what must not change, and what fills the vacated spot. Only the grip sentence
# differs between them.
OPENING = (
    "Add a single adult man's bare right hand and forearm to the image, reaching in from "
    "the bottom right of the frame, and pick the object up off the counter with it. "
)
CLOSING = (
    " The object is lifted well clear of the countertop, raised about 20 to 30 cm above "
    "it, fully supported by the hand and no longer touching the counter. Keep the object "
    "exactly the same as it was — the same size, shape, proportions, colour, material, "
    "surface texture, markings and wear. Keep the same kitchen counter and the same "
    "background, and the spot on the counter where the object was resting is now empty "
    "bare countertop. The forearm is bare to just below the elbow and is cut off by the "
    "bottom edge of the frame — the hand and forearm are the only part of a person in "
    "the picture. Natural indoor lighting, sharp focus on the hand and the object, "
    "realistic photo."
)

# Object-agnostic grip description for the "add" variant. Mirrors the five slots
# of a grasp spec (GRASP_GUIDE.md) but fills each abstractly, deferring to the
# object visible in the frame.
GENERIC_GRIP = (
    "The hand holds the object in the functional grip a person actually uses when using "
    "that object for its purpose — the working grip for the task the object is for, not "
    "just a pinch to move it aside — with the fingers and thumb wrapped or placed on the "
    "part of the object that grip requires, and the object held in the orientation it "
    "would be used in, as if about to be used."
)


def variants_for(obj_name: str) -> dict[str, str]:
    _, grasp = GRASPS[obj_name]
    return {
        "add": OPENING + GENERIC_GRIP + CLOSING,
        "spec": OPENING + grasp + "." + CLOSING,
    }


def default_sources(obj_name: str) -> list[Path]:
    return sorted((PAIRS_ROOT / obj_name).glob(f"{obj_name}_grasp*_seed*_before.png"))


# Not [0, 1]: the source images were themselves rendered with seeds 0 and 1 at
# these exact dims, so reusing a source's own seed hands the sampler the very
# noise tensor the condition image was denoised from, and the counter comes out
# covered in dense black speckle. Measured on all 64 renders of the first run:
# every seed-collision image had a counter-region Laplacian variance of 300-2100
# vs 15-55 for every non-collision image. Any seeds work as long as they differ
# from the seed embedded in the source filename.
SEEDS = [2, 3]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", action="append", help="source on-counter image (repeatable)")
    ap.add_argument("--object", dest="obj_name", default="pizza-peel", choices=sorted(GRASPS))
    ap.add_argument("--out", help="output subdirectory under outputs/")
    args = ap.parse_args()

    sources = ([Path(p) if Path(p).is_absolute() else ROOT / p for p in args.input]
               if args.input else default_sources(args.obj_name))
    if not sources:
        raise SystemExit(f"no source images for {args.obj_name} under {PAIRS_ROOT}")
    for p in sources:
        if not p.exists():
            raise SystemExit(f"missing source image: {p}")

    variants = variants_for(args.obj_name)
    out_dir = ROOT / "outputs" / (args.out or f"pickup_{args.obj_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    # As in run_place_down.py: let the pipeline derive dims from the condition
    # image. Forcing dims that disagree with the derived ones shreds the counter
    # into speckle.
    t0 = time.time()
    pipe = load(quiet=True)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = [(src, vname, prompt, seed)
            for src in sources
            for vname, prompt in variants.items()
            for seed in SEEDS]

    results = []
    for i, (src, vname, prompt, seed) in enumerate(jobs, 1):
        image = Image.open(src).convert("RGB")
        # knife_grasp0_seed1_before -> grasp0_seed1
        stem = src.stem.removeprefix(f"{args.obj_name}_").removesuffix("_before")
        name = f"{args.obj_name}--{stem}--{vname}--seed{seed}"
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
            )
        elapsed = time.time() - t0

        dest = out_dir / f"{name}.png"
        result = out.images[0]
        result.save(dest)

        # before | after: source (on counter) left, generated grasp right.
        pair_dest = pairs_dir / f"{name}--pair.png"
        hstack(image, result).save(pair_dest)

        print(f" {elapsed:.1f}s", flush=True)
        results.append(
            {"object": args.obj_name, "source": str(src.relative_to(ROOT)), "source_key": stem,
             "variant": vname, "seed": seed, "prompt": prompt, "output": dest.name,
             "pair": str(pair_dest.relative_to(out_dir)), "seconds": round(elapsed, 1),
             "size": list(result.size)}
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_review(results, args.obj_name, variants, out_dir)
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} images in {total / 60:.1f}min. Review: {out_dir / 'review.html'}")


def write_review(results: list[dict], obj_name: str, variants: dict[str, str],
                 out_dir: Path) -> None:
    """One section per source, one row per variant, each cell a before|after pair."""
    sections = []
    for src in dict.fromkeys(r["source"] for r in results):
        rows = []
        for vname in variants:
            cells = "".join(
                f"""<figure><img src="{html.escape(r['pair'])}">
                <figcaption>seed {r['seed']} &middot; {r['seconds']}s
                &middot; before | after</figcaption></figure>"""
                for r in results
                if r["source"] == src and r["variant"] == vname
            )
            rows.append(f"""
      <h3>{html.escape(vname)}</h3>
      <p class="prompt">{html.escape(variants[vname])}</p>
      <div class="grid">{cells}</div>""")
        sections.append(f"""
  <section>
    <h2>source: {html.escape(Path(src).name)}</h2>
    {''.join(rows)}
  </section>""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>picked up &mdash; {html.escape(obj_name)}</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font: 15px/1.55 system-ui, sans-serif; margin: 0 auto; padding: 2rem; max-width: 1500px;
         background: #14161a; color: #e8e8ea; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .meta {{ color: #9aa0a8; margin: 0 0 1rem; font-size: .875rem; }}
  .neg {{ padding: .6rem .8rem; background: #2a1c1c; border-left: 3px solid #a05050; border-radius: 3px;
          font-size: .8125rem; margin: 0 0 2rem; }}
  section {{ border-top: 1px solid #2a2e35; padding: 1.5rem 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .75rem; font-family: ui-monospace, monospace; color: #8fd3ff; }}
  h3 {{ font-size: .9rem; margin: 1.5rem 0 .4rem; font-family: ui-monospace, monospace; color: #ffcf8f; }}
  .prompt {{ margin: 0 0 .75rem; padding: .5rem .7rem; background: #1c1f25; border-left: 3px solid #4a5058;
             border-radius: 3px; font-size: .8125rem; }}
  .grid {{ display: grid; grid-template-columns: 1fr; gap: .75rem; }}
  figure {{ margin: 0; }}
  img {{ width: 100%; height: auto; border-radius: 4px; display: block; background: #000; }}
  figcaption {{ font-size: .75rem; color: #9aa0a8; margin-top: .3rem; font-family: ui-monospace, monospace; }}
</style>
<h1>Picked up &mdash; {html.escape(obj_name)} (hand added, functional grasp)</h1>
<p class="meta">{len(results)} images &middot; {len(variants)} prompt variants &times; {len(SEEDS)} seeds
 per source &middot; {results[0]['size'][0]}&times;{results[0]['size'][1]}
 &middot; 40 steps, true_cfg_scale 4.0, bf16</p>
<p class="neg"><b>negative_prompt:</b> {html.escape(NEGATIVE)}</p>
{''.join(sections)}
"""
    (out_dir / "review.html").write_text(doc)


if __name__ == "__main__":
    main()
