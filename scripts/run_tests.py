"""Run a spread of image-edit scenarios and build an HTML review sheet.

Loads the 20B pipeline once and reuses it across every case, so the ~1min model
load is paid a single time.
"""

from __future__ import annotations

import html
import json
import time
from pathlib import Path

import torch
from PIL import Image

from pipeline import DEFAULTS, load

ROOT = Path(__file__).resolve().parent.parent
INPUTS = ROOT / "inputs"
OUTPUTS = ROOT / "outputs"

# Each case probes a different capability of the edit model.
CASES = [
    {
        "name": "01-add-object",
        "probes": "Local object insertion — does it add the hat without repainting the cat?",
        "images": ["cat.png"],
        "prompt": "Add a small blue wizard hat with gold stars on the cat's head. Keep everything else unchanged.",
    },
    {
        "name": "02-replace-background",
        "probes": "Background swap with subject identity preserved.",
        "images": ["cat.png"],
        "prompt": (
            "Replace the green wall behind the cat with a snowy pine forest at dusk. "
            "Keep the cat's fur pattern, pose and the wooden ledge exactly the same, "
            "and relight the cat to match the cool blue ambient light."
        ),
    },
    {
        "name": "03-render-text",
        "probes": "Text rendering in-scene — Qwen's headline capability.",
        "images": ["inpaint.png"],
        "prompt": (
            'Add a weathered brown wooden national-park sign at the right side of the road, '
            'with the text "HALF DOME OVERLOOK" carved and painted in white on it. '
            "Match the scene's perspective and lighting."
        ),
    },
    {
        "name": "04-style-transfer",
        "probes": "Global restyle while holding composition.",
        "images": ["inpaint.png"],
        "prompt": (
            "Restyle this landscape as a traditional Japanese ukiyo-e woodblock print with flat "
            "colour areas, visible woodgrain texture and bold outlines. Keep the exact same "
            "composition, mountain shape and tree placement."
        ),
    },
    {
        "name": "05-multi-image-compose",
        "probes": "Two-image composition — transplant a subject from image 2 into image 1's scene.",
        "images": ["astronaut.jpg", "cat.png"],
        "prompt": (
            "Place the black-and-white cat from the second image onto the lunar surface next to "
            "the little astronaut figurine in the first image. The cat sits on the grey moon dust "
            "on the right, looking at the astronaut. Keep the first image's dark sky, shallow "
            "depth of field and dramatic lighting."
        ),
    },
    {
        "name": "06-lighting-season",
        "probes": "Global lighting / season change without geometry drift.",
        "images": ["inpaint.png"],
        "prompt": (
            "Change this to a winter scene at golden-hour sunset: snow on the mountain and the "
            "road, warm orange light on the rock face, long shadows from the pines. "
            "Do not move the mountain or the trees."
        ),
    },
    {
        "name": "07-object-removal",
        "probes": "Removal + plausible inpainting of the revealed area.",
        "images": ["i2i.png"],
        "prompt": (
            "Remove the two smaller astronaut figurines on the left and right, leaving only the "
            "central astronaut standing on the wooden table. Fill in the vacated space with the "
            "same blurred bookshelf background and wooden tabletop."
        ),
    },
    {
        "name": "08-viewpoint-change",
        "probes": "Novel view synthesis — the hardest ask; tests 3D consistency.",
        "images": ["i2i.png"],
        "prompt": (
            "Show the same three astronaut figurines on the same table, but from a low "
            "eye-level angle close to the tabletop, looking slightly upward at them."
        ),
    },
]


def main() -> None:
    OUTPUTS.mkdir(exist_ok=True)

    t_load = time.time()
    pipe = load(quiet=True)
    load_s = time.time() - t_load
    print(f"pipeline loaded in {load_s:.1f}s", flush=True)

    results = []
    for i, case in enumerate(CASES, 1):
        images = [Image.open(INPUTS / n).convert("RGB") for n in case["images"]]
        print(f"[{i}/{len(CASES)}] {case['name']} ...", end="", flush=True)

        t0 = time.time()
        with torch.inference_mode():
            out = pipe(
                image=images,
                prompt=case["prompt"],
                negative_prompt=DEFAULTS["negative_prompt"],
                true_cfg_scale=DEFAULTS["true_cfg_scale"],
                num_inference_steps=DEFAULTS["num_inference_steps"],
                guidance_scale=DEFAULTS["guidance_scale"],
                num_images_per_prompt=1,
                generator=torch.manual_seed(0),
            )
        elapsed = time.time() - t0

        dest = OUTPUTS / f"{case['name']}.png"
        result = out.images[0]
        result.save(dest)
        print(f" {elapsed:.1f}s -> {dest.name} ({result.size[0]}x{result.size[1]})", flush=True)

        results.append({**case, "output": dest.name, "seconds": round(elapsed, 1), "size": list(result.size)})

    (OUTPUTS / "results.json").write_text(json.dumps(results, indent=2))
    write_review(results)
    total = sum(r["seconds"] for r in results)
    print(f"\n{len(results)} edits in {total:.0f}s ({total / len(results):.0f}s avg). "
          f"Review: {OUTPUTS / 'review.html'}")


def write_review(results: list[dict]) -> None:
    """Emit a side-by-side input/output sheet for eyeballing the edits."""
    rows = []
    for r in results:
        ins = "".join(
            f'<figure><img src="../inputs/{html.escape(n)}"><figcaption>input {j}: {html.escape(n)}</figcaption></figure>'
            for j, n in enumerate(r["images"], 1)
        )
        rows.append(f"""
  <section>
    <h2>{html.escape(r['name'])}</h2>
    <p class="probes">{html.escape(r['probes'])}</p>
    <p class="prompt">{html.escape(r['prompt'])}</p>
    <div class="pair">
      <div class="side">{ins}</div>
      <div class="arrow">&rarr;</div>
      <div class="side">
        <figure><img src="{html.escape(r['output'])}">
        <figcaption>output &middot; {r['size'][0]}&times;{r['size'][1]} &middot; {r['seconds']}s</figcaption></figure>
      </div>
    </div>
  </section>""")

    total = sum(r["seconds"] for r in results)
    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Qwen-Image-Edit-2511 &mdash; test results</title>
<style>
  :root {{ color-scheme: dark light; }}
  body {{ font: 15px/1.55 system-ui, sans-serif; margin: 0 auto; padding: 2rem; max-width: 1200px;
         background: #14161a; color: #e8e8ea; }}
  h1 {{ font-size: 1.5rem; margin: 0 0 .25rem; }}
  .meta {{ color: #9aa0a8; margin: 0 0 2rem; font-size: .875rem; }}
  section {{ border-top: 1px solid #2a2e35; padding: 1.5rem 0; }}
  h2 {{ font-size: 1.05rem; margin: 0 0 .35rem; font-family: ui-monospace, monospace; color: #8fd3ff; }}
  .probes {{ margin: 0 0 .5rem; color: #9aa0a8; font-size: .875rem; }}
  .prompt {{ margin: 0 0 1rem; padding: .6rem .8rem; background: #1c1f25; border-left: 3px solid #4a5058;
             border-radius: 3px; font-size: .875rem; }}
  .pair {{ display: flex; align-items: center; gap: 1rem; }}
  .side {{ display: flex; gap: .75rem; flex: 1; }}
  .arrow {{ color: #6a7078; font-size: 1.5rem; flex: none; }}
  figure {{ margin: 0; flex: 1; min-width: 0; }}
  img {{ width: 100%; height: auto; border-radius: 4px; display: block; background: #000; }}
  figcaption {{ font-size: .75rem; color: #9aa0a8; margin-top: .3rem; font-family: ui-monospace, monospace; }}
</style>
<h1>Qwen-Image-Edit-2511 &mdash; image editing tests</h1>
<p class="meta">{len(results)} cases &middot; 40 steps, true_cfg_scale 4.0, bf16, seed 0, single H200
 &middot; {total:.0f}s total, {total / len(results):.0f}s/edit</p>
{''.join(rows)}
"""
    (OUTPUTS / "review.html").write_text(doc)


if __name__ == "__main__":
    main()
