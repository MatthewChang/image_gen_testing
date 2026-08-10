# image_gen — Qwen-Image-Edit-2511

Pixi environment for instruction-based image editing with
[Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
(20B, bf16, Apache-2.0) via `diffusers`' `QwenImageEditPlusPipeline`.

## Setup

```bash
pixi install     # ~3 min
pixi run download   # ~54GB snapshot into ./.hf (one time)
pixi run gpu-check  # confirm torch/CUDA/pipeline import
```

The model card requires `diffusers` from git — `QwenImageEditPlusPipeline` is not
in any tagged release yet. `HF_HOME` is pinned to `./.hf` so the weights live on
the shared volume next to the project rather than in the container's `~/.cache`.

## Usage

```bash
# single-image edit
pixi run edit -i inputs/cat.png -p "add a red bow tie" -o outputs/bowtie.png

# multi-image composition (repeat -i; images are referenced in prompt order)
pixi run edit -i inputs/astronaut.jpg -i inputs/cat.png \
  -p "place the cat from the second image on the moon beside the astronaut" \
  -o outputs/compose.png

# flags: -n/--steps (40), --cfg (true_cfg_scale, 4.0), --seed (0), --negative
```

### Output aspect ratio

By default the pipeline derives the output ratio from the *first input image*
(`calculate_dimensions(1024*1024, w/h)`), so a 4000×3000 photo yields 4:3. Pass
`--ratio` to override it — dimensions are snapped to multiples of 16 (the
transformer's 2×2 latent patching) at ~1MP:

```bash
pixi run edit -i inputs/counter.jpeg -p "..." -o out.png --ratio 16:9   # 1280x720
```

| `--ratio` | output | exact? |
|---|---|---|
| `16:9` | 1280×720 | yes |
| `4:3` | 1152×864 | yes |
| `1:1` | 1024×1024 | yes |
| `3:2` | 1248×832 | yes |
| `21:9` | 1568×672 | yes |
| `9:16` | 720×1280 | yes |
| `2.39` | 1568×656 | ±0.0002 |

The input is pre-cropped to the same ratio first, because the condition image is
encoded at its *own* aspect ratio — feeding 4:3 while asking for 16:9 makes the
model reframe content it was never shown. `--anchor top` (default) keeps the
upper strip, which on a downward-facing shot holds the scene-anchoring
background; `--anchor center` discards it and the model reinvents the room.
`--no-crop` skips the pre-crop entirely and also works well on this input.

`pixi run tests` re-runs the 8-case suite in `scripts/run_tests.py` and rebuilds
`outputs/review.html`.

### Grasp / place-down pair

Two scripts produce the two halves of a pickup, on `inputs/counter.jpeg`:

```bash
python scripts/run_grasp.py --object mug --ratio 16:9   # object held in a functional grasp
python scripts/run_place_down.py --object mug           # same object set back down, hand gone
```

`run_grasp.py` carries a per-object grasp spec in its `GRASPS` dict.

[GRASP_GUIDE.md](GRASP_GUIDE.md) — **adding an object** to `GRASPS`, or
**diagnosing a grasp render** that came out wrong.

`run_place_down.py` is object-agnostic: its prompts say "the object" and never
name it, so one prompt serves every object and grasp type. The source image
already shows the model what the object is. Resting pose is expressed as a rule
("upright on its base if it stands, on its side or flat face if it lies") rather
than a fixed pose, so a mug stands and a knife lies flat. Each render is also
written beside its source as a before|after pair under `pairs/` for judging
frame-to-frame consistency.

Note the asymmetry in `--ratio`: use it for `run_grasp.py` (input is pre-cropped
to match), but *not* for `run_place_down.py`. Forcing dims that differ from the
ones the pipeline derives from the condition image misaligns the noise and
conditioning latents, and the countertop shreds into black speckle. A 1280×720
source auto-derives to 1376×768; crop afterwards if exact pairing is needed.

## GPU notes

The whole pipeline is resident in bf16 on one H200 (~40GB of 143GB), so no
offloading or sharding. Pick a free GPU with `CUDA_VISIBLE_DEVICES=N` — GPU 0 on
this box is frequently in use by other jobs.

Measured on one idle H200: 17s model load, ~32s per 1024×1024 edit at 40 steps,
~56s for a two-image composition. Output resolution follows the first input's
aspect ratio (~1MP), not its exact pixel dimensions, unless `--ratio` overrides
it.

`guidance_scale is passed as 1.0, but ignored...` on every call is expected —
this checkpoint is not guidance-distilled and uses `true_cfg_scale` instead.
