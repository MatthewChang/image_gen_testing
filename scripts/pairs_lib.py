"""Manifest layer + shared generation core for the run_pair_* scripts.

A manifest is a CSV with header id,object_description,grasp_description,seed.
grasp_description and seed are optional columns; seed defaults to 1 (seed 0 at
the production dims reinvents the room; see docs/experiments). Ids become
directory names under the output root, so they must be unique. Running as a
script validates a manifest:

    python scripts/pairs_lib.py manifest.csv [--require-grasp]
"""

from __future__ import annotations

import argparse
import csv
import importlib
import multiprocessing as mp
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from pipeline import DEFAULTS, crop_to_ratio, load

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
                seed = int(raw_seed) if raw_seed else 1
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Validate a pair-generation manifest.")
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--require-grasp", action="store_true",
                    help="also require a non-empty grasp_description per row (pipeline A)")
    args = ap.parse_args()
    n = len(read_manifest(args.manifest, require_grasp=args.require_grasp))
    print(f"{args.manifest}: {n} row(s) OK")
