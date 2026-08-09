"""Export the "remove"-variant before/after images as flat sibling files.

Each placed_*/results.json row records which grasp frame it was edited from, so
the pairing is read from there rather than reconstructed from filenames.

Naming needs two seed slots, not one. A placed filename like
mug--seed0--remove--seed1.png carries *two* independent seeds: the seed of the
grasp frame it was edited from, and the seed of the placement render itself. So
the two grasp frames per object become grasp0/grasp1 and the two placement
renders become seed0/seed1, giving 4 pairs per object from 2 distinct after-images.

The after-image is therefore copied twice per grasp frame, once into each
placement seed's pair, so that every before/after pair stands on its own.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
DEST_ROOT = OUTPUTS / "remove_pairs"

VARIANT = "remove"


def main() -> None:
    results_files = sorted(OUTPUTS.glob("placed_*/results.json"))
    if not results_files:
        raise SystemExit(f"no placed_*/results.json under {OUTPUTS}")

    total = 0
    for results_file in results_files:
        src_dir = results_file.parent
        rows = [r for r in json.loads(results_file.read_text()) if r["variant"] == VARIANT]
        if not rows:
            print(f"{src_dir.name}: no '{VARIANT}' rows, skipped")
            continue

        obj = rows[0]["object"]
        dest_dir = DEST_ROOT / obj
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Stable grasp0/grasp1 indices, ordered by source path so the numbering
        # matches the source frames' own seed order.
        sources = sorted({r["source"] for r in rows})
        grasp_index = {s: i for i, s in enumerate(sources)}

        for r in rows:
            gi = grasp_index[r["source"]]
            stem = f"{obj}_grasp{gi}_seed{r['seed']}"

            before_src = src_dir / r["output"]          # placed down, hand gone
            after_src = ROOT / r["source"]              # held in a functional grasp
            before_dest = dest_dir / f"{stem}_before{before_src.suffix}"
            after_dest = dest_dir / f"{stem}_after{after_src.suffix}"

            # copy2 preserves the bytes exactly - no re-encode, no quality loss.
            shutil.copy2(before_src, before_dest)
            shutil.copy2(after_src, after_dest)
            total += 2

        print(f"{obj}: {len(rows)} pairs -> {dest_dir.relative_to(ROOT)}")

    print(f"\n{total} files in {DEST_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
