# Manifest-driven before/after pair pipelines

**Spec**: Two production scripts sharing a library, replacing the research sweeps with
their winning prompt variants. `run_pair_grasp_first.py` (A): functional-grasp render
(`run_grasp` `explicit`) → "remove" placing edit (`run_place_down` `remove`); grasp
frame = `after.png`, placed frame = `before.png`. `run_pair_before_first.py` (B):
object-on-counter render (`run_counter` `reach`) → "add" pickup edit (`run_pickup`
`add`); counter frame = `before.png`, pickup frame = `after.png`. Both: `-i
manifest.csv -o out_dir/`, one shared CSV schema (`id,object_description,
grasp_description,seed`; seed optional default 0; B ignores the grasp column), output
`<out>/<id>/before.png|after.png`, both frames at matching 1376×768, stage-2 seed =
row seed + 1000, idempotent skip when both frames exist (`--force` overrides),
dynamic fan-out across visible GPUs (one worker + pipeline per GPU), per-row
skip-and-continue with end-of-run failure summary and nonzero exit, images only.
Backdrop hard-coded to `inputs/counter.jpeg`. Plus a user-invoked project skill
`author-pair-manifest` that turns a `class,count` CSV into a manifest (ids =
8-char uuid hex + `_` + class slug, per-class incrementing seeds, five-slot grasp
specs per GRASP_GUIDE.md), default destination `inputs/manifests/`.

Grounding facts the code below relies on (verified):
- `pipeline.load(device="cuda", quiet=True)` moves the whole pipe to `device`;
  `DEFAULTS` carries `true_cfg_scale=4.0`, `num_inference_steps=40`,
  `guidance_scale=1.0`; `crop_to_ratio(image, ratio, anchor="top")` exists.
- The diffusers pipeline derives output dims from the condition image via
  `calculate_dimensions(1024*1024, w/h)` which rounds to multiples of 32. A
  1376×768 condition derives to exactly 1376×768, and the top-cropped 4000×3000
  backdrop (→ 4000×2233 at ratio 1376/768) also derives to 1376×768 — so both
  stages run at identical dims and requested dims never disagree with derived
  dims (the disagreement documented in `run_place_down.py` as the speckle cause).
- Reusing a source image's own generation seed at matching dims shreds the
  counter into speckle (measured in the run_pickup runs: counter-region Laplacian
  variance 300–2100 vs 15–55 clean) — hence the stage-2 seed offset.
- `run_grasp.VARIANTS["explicit"]` takes `{obj}` and `{grasp}`;
  `run_counter.VARIANTS["reach"]` takes `{obj}`;
  `run_place_down.VARIANTS["remove"]` is object-agnostic (no placeholders);
  `run_pickup.OPENING + run_pickup.GENERIC_GRIP + run_pickup.CLOSING` is the
  object-agnostic "add" prompt. Each script's `NEGATIVE` is module-level. All
  four modules are import-safe (`main()` behind `__main__` guards).

## Stage 1: `pairs_lib.py` — manifest schema, validation, CLI validator

**Files**: `scripts/pairs_lib.py` (new)

**Change**: create the file with the manifest layer only (generation lands in
Stage 2 of this plan; the module is complete and runnable at every stage).

```python
"""Manifest layer + shared generation core for the run_pair_* scripts.

A manifest is a CSV with header id,object_description,grasp_description,seed.
grasp_description and seed are optional columns; seed defaults to 0. Ids become
directory names under the output root, so they must be unique. Running as a
script validates a manifest:

    python scripts/pairs_lib.py manifest.csv [--require-grasp]
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REQUIRED_COLUMNS = {"id", "object_description"}


@dataclass(frozen=True)
class Row:
    id: str
    object_description: str
    grasp_description: str  # "" when the column is absent or empty
    seed: int


def read_manifest(path: Path, require_grasp: bool) -> list[Row]:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        cols = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - cols
        if missing:
            raise SystemExit(f"{path}: missing column(s): {', '.join(sorted(missing))}")
        if require_grasp and "grasp_description" not in cols:
            raise SystemExit(f"{path}: this pipeline needs a grasp_description column")
        rows = []
        for lineno, rec in enumerate(reader, 2):
            rid = (rec["id"] or "").strip()
            obj = (rec["object_description"] or "").strip()
            grasp = (rec.get("grasp_description") or "").strip()
            if not rid or not obj:
                raise SystemExit(f"{path}:{lineno}: id and object_description must be non-empty")
            if require_grasp and not grasp:
                raise SystemExit(f"{path}:{lineno}: empty grasp_description ({rid})")
            raw_seed = (rec.get("seed") or "").strip()
            try:
                seed = int(raw_seed) if raw_seed else 0
            except ValueError:
                raise SystemExit(f"{path}:{lineno}: seed must be an integer, got {raw_seed!r}")
            rows.append(Row(rid, obj, grasp, seed))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    ids = [r.id for r in rows]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise SystemExit(f"{path}: duplicate id(s): {', '.join(dupes)}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate a pair-generation manifest.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--require-grasp", action="store_true",
                    help="also require a non-empty grasp_description per row (pipeline A)")
    args = ap.parse_args()
    n = len(read_manifest(args.manifest, require_grasp=args.require_grasp))
    print(f"{args.manifest}: {n} row(s) OK")
```

**Validation**:

```bash
cd /k8s-nfs/dev_drmchang/projects/image_gen
printf '%s\n' 'id,object_description,grasp_description,seed' \
  'a1b2c3d4_mug,"a plain ceramic mug","hook grip, fingers through the handle",0' \
  'e5f6a7b8_mug,"a plain ceramic mug","hook grip, fingers through the handle",1' \
  > /tmp/m_ok.csv
.pixi/envs/default/bin/python scripts/pairs_lib.py /tmp/m_ok.csv --require-grasp
# expected: "/tmp/m_ok.csv: 2 row(s) OK", exit 0

printf '%s\n' 'id,object_description' 'x_knife,"a kitchen knife"' > /tmp/m_b.csv
.pixi/envs/default/bin/python scripts/pairs_lib.py /tmp/m_b.csv
# expected: "1 row(s) OK" (grasp column optional without --require-grasp)
.pixi/envs/default/bin/python scripts/pairs_lib.py /tmp/m_b.csv --require-grasp
# expected: exit 1, "needs a grasp_description column"

sed 's/e5f6a7b8_mug/a1b2c3d4_mug/' /tmp/m_ok.csv > /tmp/m_dup.csv
.pixi/envs/default/bin/python scripts/pairs_lib.py /tmp/m_dup.csv
# expected: exit 1, "duplicate id(s): a1b2c3d4_mug"
```

## Stage 2: generation core + GPU fan-out + `run_pair_grasp_first.py`

**Files**: `scripts/pairs_lib.py` (extend), `scripts/run_pair_grasp_first.py` (new)

**Change 1** — append to `scripts/pairs_lib.py` (new imports merge into the header:
`importlib`, `multiprocessing as mp`, `sys`, `time`, `Callable` from `typing`,
`Image` from PIL, and `from pipeline import DEFAULTS, crop_to_ratio, load`;
`torch` is imported lazily inside functions so the CLI validator stays instant):

```python
BACKDROP = ROOT / "inputs" / "counter.jpeg"

# Both frames render at exactly these dims. 1376x768 is its own fixed point of
# the pipeline's calculate_dimensions (rounds to 32 near 1MP), so stage 2 —
# which derives dims from the stage-1 frame — comes back at the same size, and
# requested dims never disagree with derived dims (that disagreement shreds the
# counter into speckle; see run_place_down.py).
WIDTH, HEIGHT = 1376, 768

# Stage 2 edits the stage-1 render at the same dims, so reusing the stage-1
# seed would hand the sampler the very noise tensor the condition image was
# denoised from — the same speckle failure, measured in run_pickup.py's first
# run. Any constant offset avoids the collision.
STAGE2_SEED_OFFSET = 1000

# A worker that dies without reporting (e.g. host OOM kill) would block the
# parent forever; a whole pair is ~70s, so half an hour means something is gone.
RESULT_TIMEOUT_S = 1800


@dataclass(frozen=True)
class PairSpec:
    """Everything that distinguishes the two pipelines: the two prompts, their
    negatives, and which chained render is the before frame."""
    stage1_prompt: Callable[[Row], str]
    stage1_negative: str
    stage1_file: str  # "before.png" or "after.png"
    stage2_prompt: Callable[[Row], str]
    stage2_negative: str
    stage2_file: str


def _render(pipe, image, prompt: str, negative: str, seed: int, fix_size: bool):
    import torch

    size = {"width": WIDTH, "height": HEIGHT} if fix_size else {}
    with torch.inference_mode():
        out = pipe(
            image=[image],
            prompt=prompt,
            negative_prompt=negative,
            true_cfg_scale=DEFAULTS["true_cfg_scale"],
            num_inference_steps=DEFAULTS["num_inference_steps"],
            guidance_scale=DEFAULTS["guidance_scale"],
            num_images_per_prompt=1,
            generator=torch.manual_seed(seed),
            **size,
        )
    return out.images[0]


def generate_pair(pipe, row: Row, row_dir: Path, spec: PairSpec) -> None:
    row_dir.mkdir(parents=True, exist_ok=True)
    backdrop = crop_to_ratio(Image.open(BACKDROP).convert("RGB"), WIDTH / HEIGHT, anchor="top")
    stage1 = _render(pipe, backdrop, spec.stage1_prompt(row), spec.stage1_negative,
                     row.seed, fix_size=True)
    stage1.save(row_dir / spec.stage1_file)
    stage2 = _render(pipe, stage1, spec.stage2_prompt(row), spec.stage2_negative,
                     row.seed + STAGE2_SEED_OFFSET, fix_size=False)
    stage2.save(row_dir / spec.stage2_file)


def _pair_exists(row_dir: Path) -> bool:
    return (row_dir / "before.png").exists() and (row_dir / "after.png").exists()


def _worker(module_name: str, gpu: int, job_q, result_q, out_dir_str: str) -> None:
    """One process per GPU. The spec is reached by importing the calling script
    by name (spawn re-imports cleanly; lambdas in a spec never cross processes)."""
    mod = importlib.import_module(module_name)
    pipe = load(device=f"cuda:{gpu}", quiet=True)
    print(f"[gpu{gpu}] pipeline loaded", flush=True)
    out_dir = Path(out_dir_str)
    while True:
        row = job_q.get()
        if row is None:
            return
        t0 = time.time()
        try:
            generate_pair(pipe, row, out_dir / row.id, mod.SPEC)
        except Exception as e:  # a bad row must not kill the worker
            result_q.put((row.id, f"{type(e).__name__}: {e}"))
            print(f"[gpu{gpu}] {row.id} FAILED: {e}", flush=True)
        else:
            result_q.put((row.id, None))
            print(f"[gpu{gpu}] {row.id} {time.time() - t0:.0f}s", flush=True)


def main_cli(module_name: str, require_grasp: bool, description: str) -> None:
    import torch

    ap = argparse.ArgumentParser(
        description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--manifest", required=True, type=Path)
    ap.add_argument("-o", "--out", required=True, type=Path,
                    help="output root: one <id>/before.png + <id>/after.png per row. "
                         "Give each pipeline its own root — both write the same "
                         "filenames, so a shared root would collide on shared ids.")
    ap.add_argument("--force", action="store_true",
                    help="regenerate rows whose pair already exists")
    args = ap.parse_args()

    rows = read_manifest(args.manifest, require_grasp=require_grasp)
    pending = [r for r in rows if args.force or not _pair_exists(args.out / r.id)]
    if len(pending) < len(rows):
        print(f"{len(rows) - len(pending)} row(s) already complete, skipped")
    if not pending:
        return

    n_gpu = torch.cuda.device_count()
    if n_gpu == 0:
        raise SystemExit("no CUDA devices visible")
    n_workers = min(n_gpu, len(pending))
    print(f"{len(pending)} row(s) across {n_workers} GPU(s)")

    ctx = mp.get_context("spawn")
    job_q, result_q = ctx.Queue(), ctx.Queue()
    for r in pending:
        job_q.put(r)
    for _ in range(n_workers):
        job_q.put(None)  # one sentinel per worker
    workers = [
        ctx.Process(target=_worker, args=(module_name, g, job_q, result_q, str(args.out)),
                    daemon=True)
        for g in range(n_workers)
    ]
    for w in workers:
        w.start()

    failures = []
    for _ in pending:
        try:
            rid, err = result_q.get(timeout=RESULT_TIMEOUT_S)
        except Exception:
            for w in workers:
                w.terminate()
            raise SystemExit("a worker went silent; rerun to resume (completed pairs are skipped)")
        if err:
            failures.append((rid, err))
    for w in workers:
        w.join()

    print(f"\n{len(pending) - len(failures)}/{len(pending)} pairs -> {args.out}")
    if failures:
        for rid, err in failures:
            print(f"FAILED {rid}: {err}", file=sys.stderr)
        raise SystemExit(1)
```

The `if __name__ == "__main__":` validator block from Stage 1 stays at the bottom
of the file, unchanged.

**Change 2** — `scripts/run_pair_grasp_first.py` (new, complete):

```python
"""Grasp-first pairs: functional-grasp render, then the "remove" placing edit.

Every manifest row needs id, object_description and grasp_description (see
pairs_lib.read_manifest). The grasp render is the *after* frame; the placing
edit of it is the *before* frame:

    <out>/<id>/after.png    object held in a functional grasp (run_grasp "explicit")
    <out>/<id>/before.png   same object set down, hand gone (run_place_down "remove")

Prompts are the winning variants of the run_grasp.py / run_place_down.py
sweeps, imported from those scripts rather than copied.
"""

from __future__ import annotations

import run_grasp
import run_place_down
from pairs_lib import PairSpec, main_cli

# Workers import this module by name to reach SPEC (see pairs_lib._worker).
MODULE = "run_pair_grasp_first"

SPEC = PairSpec(
    stage1_prompt=lambda row: run_grasp.VARIANTS["explicit"].format(
        obj=row.object_description, grasp=row.grasp_description),
    stage1_negative=run_grasp.NEGATIVE,
    stage1_file="after.png",
    stage2_prompt=lambda row: run_place_down.VARIANTS["remove"],
    stage2_negative=run_place_down.NEGATIVE,
    stage2_file="before.png",
)

if __name__ == "__main__":
    main_cli(MODULE, require_grasp=True, description=__doc__)
```

**Validation** (two rows exercise fan-out across two GPUs; descriptions taken from
`run_grasp.GRASPS` so quality is a known quantity):

```bash
cd /k8s-nfs/dev_drmchang/projects/image_gen
mkdir -p inputs/manifests
.pixi/envs/default/bin/python - <<'EOF'
import csv, sys
sys.path.insert(0, "scripts")
from run_grasp import GRASPS
with open("inputs/manifests/smoke.csv", "w", newline="") as f:
    w = csv.writer(f, quoting=csv.QUOTE_ALL)
    w.writerow(["id", "object_description", "grasp_description", "seed"])
    w.writerow(["deadbeef_knife", GRASPS["knife"][0], GRASPS["knife"][1], 0])
    w.writerow(["cafef00d_mug", GRASPS["mug"][0], GRASPS["mug"][1], 0])
EOF
.pixi/envs/default/bin/python scripts/run_pair_grasp_first.py \
  -i inputs/manifests/smoke.csv -o outputs/pairs_grasp_first_smoke/
# expected: two "[gpuN] <id> ...s" lines, then "2/2 pairs -> ...", exit 0
# (~3.5 min: 55s load + ~70s/pair, rows on separate GPUs)

.pixi/envs/default/bin/python - <<'EOF'
from pathlib import Path
import numpy as np
from PIL import Image
for p in sorted(Path("outputs/pairs_grasp_first_smoke").glob("*/[ba]*.png")):
    im = Image.open(p)
    a = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = a.shape
    a = a[h//3:, :3*w//4]
    lap = a[1:-1,1:-1]*4 - a[:-2,1:-1] - a[2:,1:-1] - a[1:-1,:-2] - a[1:-1,2:]
    assert im.size == (1376, 768), (p, im.size)
    assert lap.var() < 200, (p, lap.var())   # speckled images measure 300+
    print(f"{p.parent.name}/{p.name}  {im.size}  lapvar={lap.var():.0f}")
EOF
# expected: 4 lines (2 ids x before/after), all 1376x768, all lapvar < 200

# idempotency: reruns in seconds, generates nothing
.pixi/envs/default/bin/python scripts/run_pair_grasp_first.py \
  -i inputs/manifests/smoke.csv -o outputs/pairs_grasp_first_smoke/
# expected: "2 row(s) already complete, skipped", exit 0

# then eyeball outputs/pairs_grasp_first_smoke/*/  (grasp correct, hand gone in before)
```

## Stage 3: `run_pair_before_first.py`

**Files**: `scripts/run_pair_before_first.py` (new, complete)

**Change**:

```python
"""Before-first pairs: object-on-counter render, then the "add" pickup edit.

Manifest rows need only id and object_description — the pickup prompt is
object-agnostic (run_pickup's "add" variant lets the model infer the functional
grip from the object in frame), so no grasp spec is required:

    <out>/<id>/before.png   object alone on the counter (run_counter "reach")
    <out>/<id>/after.png    same object lifted in a functional grasp (run_pickup "add")

Prompts are the winning variants of the run_counter.py / run_pickup.py sweeps,
imported from those scripts rather than copied.
"""

from __future__ import annotations

import run_counter
import run_pickup
from pairs_lib import PairSpec, main_cli

# Workers import this module by name to reach SPEC (see pairs_lib._worker).
MODULE = "run_pair_before_first"

SPEC = PairSpec(
    stage1_prompt=lambda row: run_counter.VARIANTS["reach"].format(
        obj=row.object_description),
    stage1_negative=run_counter.NEGATIVE,
    stage1_file="before.png",
    stage2_prompt=lambda row: run_pickup.OPENING + run_pickup.GENERIC_GRIP
        + run_pickup.CLOSING,
    stage2_negative=run_pickup.NEGATIVE,
    stage2_file="after.png",
)

if __name__ == "__main__":
    main_cli(MODULE, require_grasp=False, description=__doc__)
```

**Validation** (same manifest as Stage 2 — B ignores the grasp column; separate
output root by design):

```bash
cd /k8s-nfs/dev_drmchang/projects/image_gen
.pixi/envs/default/bin/python scripts/run_pair_before_first.py \
  -i inputs/manifests/smoke.csv -o outputs/pairs_before_first_smoke/
# expected: "2/2 pairs -> ...", exit 0

# same dims/speckle assertions as Stage 2, pointed at the new root:
.pixi/envs/default/bin/python - <<'EOF'
from pathlib import Path
import numpy as np
from PIL import Image
for p in sorted(Path("outputs/pairs_before_first_smoke").glob("*/[ba]*.png")):
    im = Image.open(p)
    a = np.asarray(im.convert("L"), dtype=np.float32)
    h, w = a.shape
    a = a[h//3:, :3*w//4]
    lap = a[1:-1,1:-1]*4 - a[:-2,1:-1] - a[2:,1:-1] - a[1:-1,:-2] - a[1:-1,2:]
    assert im.size == (1376, 768), (p, im.size)
    assert lap.var() < 200, (p, lap.var())
    print(f"{p.parent.name}/{p.name}  {im.size}  lapvar={lap.var():.0f}")
EOF
# expected: 4 lines, all 1376x768, all lapvar < 200
# then eyeball: before = object alone near the front edge; after = one bare
# right hand lifting it in a plausible functional grip
```

## Stage 4: `author-pair-manifest` skill

**Files**: `.claude/skills/author-pair-manifest/SKILL.md` (new, in the project repo)

**Change**: user-invoked (only ever fired by hand, so it pays no context load);
the five grasp slots stay in GRASP_GUIDE.md — the skill points, never restates.

```markdown
---
name: author-pair-manifest
description: Author a pair-generation manifest from a CSV of object classes — object descriptions, five-slot grasp specs, ids and seeds.
disable-model-invocation: true
---

Turn a CSV of object classes into a manifest for the pair-generation scripts
(`scripts/run_pair_grasp_first.py` and `scripts/run_pair_before_first.py`).

**Input** ($ARGUMENTS): path to a classes CSV, optionally followed by a
destination for the manifest. The classes CSV has a header `class,count`;
`count` is optional and defaults to 1.

**Output**: one CSV with header `id,object_description,grasp_description,seed`,
quoted fields, written to the destination — default
`inputs/manifests/<input stem>_manifest.csv`. One manifest serves both
pipelines; before-first simply ignores `grasp_description`.

## Steps

1. Read the classes CSV. Where a class is ambiguous (size, material, variant —
   "knife" could be a chef's knife or a butter knife), pick the most common
   kitchen-counter interpretation and record the choice for the final report.

2. Read [GRASP_GUIDE.md](../../../GRASP_GUIDE.md), then write two descriptions
   per class:
   - `object_description` — concrete enough to render from scratch: material,
     colour, and the proportions that keep the shape honest (the guide's
     "Reading the results" section shows why shape fidelity belongs here, not
     in the grasp text).
   - `grasp_description` — fill all five slots from the guide. For part-like
     objects with no in-use grip, the guide's transport-grip rule applies.

3. Emit `count` rows per class, all sharing that class's two descriptions:
   - `id`: `uuid4().hex[:8]` + `_` + class slug (lowercase, spaces to hyphens)
   - `seed`: 0, 1, ... within each class

4. Write the manifest, then validate:

       python scripts/pairs_lib.py <manifest> --require-grasp

   Fix and re-run until it prints `N row(s) OK`.

Done when the validator passes and every grasp description fills all five
slots — reread each against the guide's slot list before reporting the
manifest path and any interpretation choices from step 1.
```

**Validation**:

```bash
cd /k8s-nfs/dev_drmchang/projects/image_gen
printf 'class,count\nspatula,2\ntennis ball,1\n' > /tmp/classes.csv
```

Then in a Claude Code session in this repo: `/author-pair-manifest /tmp/classes.csv`.
Expected: `inputs/manifests/classes_manifest.csv` with 3 rows; ids match
`^[0-9a-f]{8}_(spatula|tennis-ball)$`; spatula rows have seeds 0 and 1; the
skill's own validator run prints `3 row(s) OK`; and
`.pixi/envs/default/bin/python scripts/pairs_lib.py inputs/manifests/classes_manifest.csv --require-grasp`
reproduces that from the shell.

## Stage 5: README

**Files**: `README.md`

**Change**: replace the "Grasp / place-down pair" section's opening (the two
research-script commands stay, reframed as the sweep tooling) by inserting this
new subsection immediately after the `### Grasp / place-down pair` heading and
before the current text:

```markdown
### Manifest pipelines (production)

Two scripts generate before/after pickup pairs from a manifest CSV
(`id,object_description,grasp_description,seed`; seed optional, default 0),
writing `<out>/<id>/before.png` + `<out>/<id>/after.png`, both 1376×768:

```bash
# A: grasp render ("explicit") -> "remove" placing edit; needs grasp_description
python scripts/run_pair_grasp_first.py -i inputs/manifests/m.csv -o outputs/pairs_grasp_first/

# B: counter render ("reach") -> "add" pickup edit; object_description only
python scripts/run_pair_before_first.py -i inputs/manifests/m.csv -o outputs/pairs_before_first/
```

Rows fan out dynamically across visible GPUs (one pipeline per GPU). Reruns are
idempotent — rows whose pair exists are skipped (`--force` regenerates), so
interrupted runs resume and failures retry with the same command. Failed rows
are summarised at exit with a nonzero status. Give each pipeline its own `-o`:
they write the same filenames under the same ids.

The `/author-pair-manifest` skill authors a manifest from a `class,count` CSV,
generating descriptions, grasp specs, ids and seeds. Validate any manifest with
`python scripts/pairs_lib.py manifest.csv [--require-grasp]`.
```

Also, in the existing paragraph below it, change the sentence
`Two scripts produce the two halves of a pickup, on inputs/counter.jpeg:` to
`The underlying prompt research lives in four sweep scripts (variants × seeds, with review sheets):`
so the research/production split is explicit.

**Validation**:

```bash
grep -c "run_pair_grasp_first\|run_pair_before_first\|author-pair-manifest" \
  /k8s-nfs/dev_drmchang/projects/image_gen/README.md
# expected: >= 3
```

## Final check (done)

- Every symbol referenced exists (`load`, `DEFAULTS`, `crop_to_ratio`,
  `run_grasp.VARIANTS/NEGATIVE/GRASPS`, `run_counter.VARIANTS/NEGATIVE`,
  `run_place_down.VARIANTS/NEGATIVE`, `run_pickup.OPENING/GENERIC_GRIP/CLOSING/NEGATIVE`)
  or is defined in an earlier stage (`Row`, `read_manifest`, `PairSpec`,
  `generate_pair`, `main_cli`).
- Each stage carries a runnable validation with expected output.
- Stages are dependency-ordered: manifest layer → generation + script A →
  script B → skill (uses the Stage 1 validator) → docs.
```
