"""Invert a grasp image: set the held object down on the counter, hand gone.

Input is an output of run_grasp.py, not inputs/counter.jpeg — the point is to
produce the "before pickup" frame of *that specific object*, so the object's
identity has to survive the edit while the hand and forearm leave.

The prompts are object-agnostic by design: they say "the object", never what it
is, so the same three work for any object and any grasp type. The source image
already tells the model what the object is; naming it again only invites drift
toward a generic version of that noun. What replaced the peel-specific wording is
a rule about *resting orientation* ("upright on its base if it stands, on its
side or flat face if it lies") rather than a fixed pose, since a mug and a knife
come to rest differently.

Each render is also written out beside its source as a before|after pair under
pairs/, which is the useful artifact for judging frame-to-frame consistency.

The three variants probe one open question. In from-scratch generation naming
"hands" positively summoned them (see run_counter.py), but this checkpoint is
instruction-tuned for editing, where "remove the hand" is the idiomatic phrasing
and may well be obeyed as an instruction rather than absorbed as content. So:

  remove    - explicit edit instruction, names the hand
  restate   - describes only the target scene, never says "hand"
  before    - temporal framing, leans on the model's before/after prior

Whichever wins, the hand vocabulary stays in the negative prompt regardless.
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import DEFAULTS, dims_for_ratio, load, parse_ratio

ROOT = Path(__file__).resolve().parent.parent

# "explicit" was the strongest grasp variant, so default to its frames for
# whichever object is being run.
DEFAULT_SOURCE_SEEDS = [0, 1]


def default_sources(obj_name: str) -> list[Path]:
    return [ROOT / "outputs" / f"grasp_{obj_name}" / f"{obj_name}--explicit--seed{s}.png"
            for s in DEFAULT_SOURCE_SEEDS]

# The whole person goes, and the object must not be left floating where the hand
# used to hold it — hence the hovering terms alongside the anatomy.
NEGATIVE = (
    "hand, hands, fingers, thumb, fingertip, knuckles, arm, arms, wrist, forearm, elbow, "
    "person, people, human, body, skin, holding, gripping, grasping, lifting, reaching, "
    "floating object, hovering object, object in mid-air, object tilted at an angle, "
    "levitating, disembodied hand, blurry, deformed, low quality"
)

# Only ever substituted as "the object" / "the same object" in the prompts below —
# the entries exist so output filenames and the review sheet can name what was run,
# not because the prompt needs to know. Anything grasp-able can be added here
# without touching the prompt text.
OBJECTS = {
    "pizza-peel": "the wooden pizza peel",
    "knife": "the kitchen knife",
    "mug": "the ceramic mug",
    "mouse": "the computer mouse",
}

# The object-agnostic noun. Using "the object" rather than a description is the
# whole point: the source image already shows the model exactly what the object
# is, so naming it adds nothing and risks pulling the render toward a generic
# stock version of that noun instead of preserving the specific one in frame.
OBJ = "the object"

# Three phrasings of the same edit, all object-agnostic.
#
# Writing these to work for any object meant removing three hooks that were
# specific to the pizza peel:
#
#   "lies flat and level"     -> a mug must stand upright, not lie flat. Replaced
#                                with "in its natural resting orientation, the way
#                                it would normally sit", plus an explicit
#                                either/or, so the model picks per object.
#   "handle pointing lower right" -> not every object has a handle. Replaced with
#                                a rule about where it sits in frame, and for
#                                elongated objects only, a conditional about the
#                                long axis.
#   "same wood grain"         -> replaced with a material-neutral identity list
#                                (colour, material, texture, markings, wear).
VARIANTS = {
    # Idiomatic edit-model phrasing: name what to remove, say what fills the gap.
    # This was the strongest variant on the pizza peel.
    "remove": (
        f"Remove the hand, fingers and forearm from the image completely, and set {OBJ} down "
        f"onto the kitchen countertop. {OBJ} now rests stably on the counter surface in its "
        "natural resting orientation — the way that object would normally sit when put down "
        "and left alone, standing upright on its base if it is the sort of object that stands, "
        "or lying down on its side or its flat face if it is the sort of object that lies. It "
        "is fully supported by the counter and makes contact with it, casting a soft contact "
        "shadow directly beneath it where it touches, and it sits roughly where it was in the "
        f"frame, near the middle of the counter. Keep {OBJ} exactly the same as it was — the "
        "same size, shape, proportions, colour, material, surface texture, markings and wear. "
        "Fill in the whole area where the hand and arm used to be with more of the same "
        f"countertop surface and the same kitchen background. The counter is bare apart from {OBJ}, "
        "and nobody is in the picture."
    ),
    # Pure target-scene description — no removal verb, no anatomy named at all.
    "restate": (
        "A first person point-of-view photo from a head-mounted camera, looking down at a "
        f"kitchen counter. {OBJ} sits alone on the empty countertop, resting stably and fully "
        "supported on the counter surface in its natural resting orientation, the way it would "
        "normally sit when set down — upright on its base if it stands, or down on its side or "
        "flat face if it lies. A soft contact shadow sits directly beneath it where it meets the "
        f"counter. It is exactly the same object as before, unchanged: same size, shape, "
        "proportions, colour, material, surface texture and markings. The rest of the counter is "
        "bare, and the kitchen is empty with nobody present. Natural indoor lighting, realistic "
        "photo."
    ),
    # Temporal framing: same scene, one moment earlier. Gets "untouched" for free,
    # which does some of the work of excluding the hand without naming it.
    "before": (
        "Show this exact same kitchen counter a moment earlier, before anyone reached in or "
        f"picked anything up. {OBJ} is still sitting untouched on the countertop where it was "
        "left, resting stably and fully supported on the counter surface in its natural resting "
        "orientation — upright on its base if it is the sort of object that stands, lying on its "
        "side or flat face if it is the sort that lies — with a soft contact shadow directly "
        "beneath it. It is the same object, unchanged in size, shape, proportions, colour, "
        "material and texture, on the same counter with the same kitchen background. The room is "
        "empty, the counter is otherwise bare, and nobody has reached into the frame yet."
    ),
}

SEEDS = [0, 1]

GUTTER = 8            # px between the two frames
GUTTER_COLOUR = (24, 26, 30)


def hstack(before: Image.Image, after: Image.Image) -> Image.Image:
    """Join before|after left-to-right with a thin gutter.

    The two frames are not the same size — the grasp render is 1280x720 while the
    placed render comes back at 1376x768, since the pipeline re-derives dims from
    the condition image. Scale to a common height (the smaller of the two, so
    neither gets upsampled) rather than padding, so the object appears at
    comparable scale on both sides and inconsistencies read as inconsistencies
    rather than as a size mismatch.
    """
    h = min(before.height, after.height)
    scaled = [
        im if im.height == h
        else im.resize((round(im.width * h / im.height), h), Image.LANCZOS)
        for im in (before, after)
    ]
    total_w = sum(im.width for im in scaled) + GUTTER
    canvas = Image.new("RGB", (total_w, h), GUTTER_COLOUR)
    x = 0
    for im in scaled:
        canvas.paste(im, (x, 0))
        x += im.width + GUTTER
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-i", "--input", action="append", help="source grasp image (repeatable)")
    ap.add_argument("--object", dest="obj_name", default="pizza-peel", choices=sorted(OBJECTS))
    ap.add_argument(
        "--ratio",
        help="output aspect ratio. Default: let the pipeline derive it from the source. "
        "Overriding is usually a mistake here - see the note below.",
    )
    ap.add_argument("--out", help="output subdirectory under outputs/")
    args = ap.parse_args()

    obj = OBJECTS[args.obj_name]
    sources = ([Path(p) if Path(p).is_absolute() else ROOT / p for p in args.input]
               if args.input else default_sources(args.obj_name))
    for p in sources:
        if not p.exists():
            raise SystemExit(f"missing source image: {p}\n(run run_grasp.py --object {args.obj_name} first)")

    out_dir = ROOT / "outputs" / (args.out or f"placed_{args.obj_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_dir = out_dir / "pairs"
    pairs_dir.mkdir(exist_ok=True)

    # Deliberately NOT forcing 16:9 here, unlike run_grasp.py. A 1280x720 source
    # round-trips to 1376x768 (ratio 1.792) because the pipeline re-derives dims
    # from the condition image, and forcing 1280x720 back looks like the obvious
    # fix. It isn't: when the requested dims differ from the ones derived from the
    # condition image, the noise latents no longer align with the conditioning
    # latents, and the countertop shreds into dense black speckle (measured — every
    # forced-dim image was artifacted, every auto-dim one was clean). The 96px of
    # drift is the cheaper problem; crop afterwards if exact pairing is needed.
    size = {}
    if args.ratio:
        width, height = dims_for_ratio(parse_ratio(args.ratio))
        size = {"width": width, "height": height}
        print(f"output {width}x{height} ({args.ratio}) - forced; expect speckle artifacts", flush=True)

    t0 = time.time()
    pipe = load(quiet=True)
    print(f"pipeline loaded in {time.time() - t0:.1f}s", flush=True)

    jobs = [(src, vname, tmpl, seed)
            for src in sources
            for vname, tmpl in VARIANTS.items()
            for seed in SEEDS]

    results = []
    for i, (src, vname, tmpl, seed) in enumerate(jobs, 1):
        image = Image.open(src).convert("RGB")
        prompt = tmpl  # object-agnostic: nothing to substitute
        # The source seed disambiguates outputs when several grasp frames are used.
        stem = src.stem.rsplit("--", 1)[-1]
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
                **size,
            )
        elapsed = time.time() - t0

        dest = out_dir / f"{name}.png"
        result = out.images[0]
        result.save(dest)

        # before | after, side by side, for judging frame-to-frame consistency.
        pair_dest = pairs_dir / f"{name}--pair.png"
        pair = hstack(result, image)
        pair.save(pair_dest)

        print(f" {elapsed:.1f}s", flush=True)
        results.append(
            {"object": args.obj_name, "source": str(src.relative_to(ROOT)), "source_key": stem,
             "variant": vname, "seed": seed, "prompt": prompt, "output": dest.name,
             "pair": str(pair_dest.relative_to(out_dir)), "seconds": round(elapsed, 1),
             "size": list(result.size)}
        )

    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    write_review(results, obj, out_dir)
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} images in {total / 60:.1f}min. Review: {out_dir / 'review.html'}")


def write_review(results: list[dict], obj: str, out_dir: Path) -> None:
    """One section per source, one row per variant, each cell a before|after pair.

    Showing the pair rather than the placed frame alone is the point: the question
    is not "is this a good photo" but "is this the same object and the same
    kitchen as the frame beside it".
    """
    sections = []
    for src in dict.fromkeys(r["source"] for r in results):
        rows = []
        for vname in VARIANTS:
            cells = "".join(
                f"""<figure><img src="{html.escape(r['pair'])}">
                <figcaption>seed {r['seed']} &middot; {r['seconds']}s
                &middot; before | after</figcaption></figure>"""
                for r in results
                if r["source"] == src and r["variant"] == vname
            )
            rows.append(f"""
      <h3>{html.escape(vname)}</h3>
      <p class="prompt">{html.escape(VARIANTS[vname])}</p>
      <div class="grid">{cells}</div>""")
        sections.append(f"""
  <section>
    <h2>source: {html.escape(Path(src).name)}</h2>
    {''.join(rows)}
  </section>""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>placed down &mdash; {html.escape(obj)}</title>
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
  /* One pair per row: each cell is already two frames wide. */
  .grid {{ display: grid; grid-template-columns: 1fr; gap: .75rem; }}
  figure {{ margin: 0; }}
  img {{ width: 100%; height: auto; border-radius: 4px; display: block; background: #000; }}
  figcaption {{ font-size: .75rem; color: #9aa0a8; margin-top: .3rem; font-family: ui-monospace, monospace; }}
</style>
<h1>Placed down &mdash; {html.escape(obj)} (hand removed)</h1>
<p class="meta">{len(results)} images &middot; {len(VARIANTS)} prompt variants &times; {len(SEEDS)} seeds
 per source &middot; {results[0]['size'][0]}&times;{results[0]['size'][1]}
 &middot; 40 steps, true_cfg_scale 4.0, bf16</p>
<p class="neg"><b>negative_prompt:</b> {html.escape(NEGATIVE)}</p>
{''.join(sections)}
"""
    (out_dir / "review.html").write_text(doc)


if __name__ == "__main__":
    main()
