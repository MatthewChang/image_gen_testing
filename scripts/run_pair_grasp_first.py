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
