"""Before-first pairs: object-on-counter render, then the "add" pickup edit.

Manifest rows need only id and object_description — the pickup prompt is
object-agnostic (run_pickup's "add" variant lets the model infer the functional
grip from the object in frame), so no grasp spec is required:

    <out>/<id>/before.png   object alone on the counter (run_counter "reach")
    <out>/<id>/after.png    same object lifted in a functional grasp (run_pickup "add")

The stage-1 prompt is the winning "reach" variant of the run_counter.py sweep,
imported rather than copied. The stage-2 prompt is run_pickup's "add" skeleton
with its lift clause replaced by the "toward-camera" phrasing and its negative
extended with arm-pose terms — the winner of the lift-prompt bench
(docs/experiments/lift-prompt-iteration.md, 59/60 across 10 classes) after the
original clause proved unreliable on fresh counter renders.
"""

from __future__ import annotations

import run_counter
import run_pickup
from pairs_lib import PairSpec, main_cli

# Workers import this module by name to reach SPEC (see pairs_lib._worker).
MODULE = "run_pair_before_first"

# Anchoring the lift height to the camera ("halfway between the countertop and
# the camera") outperformed the research prompt's "20 to 30 cm" — a distance
# the model can't measure — on the lift bench; "nothing but air underneath"
# guards the graze-the-counter failure mode.
LIFT_CLAUSE = (
    " The object has been lifted up off the counter and raised toward the camera, "
    "hanging in mid-air roughly halfway between the countertop and the camera, with "
    "clear empty space visible underneath it. Nothing but air is under the object, "
    "the hand and the forearm."
)

KEEP_CLAUSE = (
    " Keep the object exactly the same as it was — the same size, shape, proportions, "
    "colour, material, surface texture, markings and wear. Keep the same kitchen counter "
    "and the same background, and the spot on the counter where the object was resting is "
    "now empty bare countertop. The forearm is bare to just below the elbow and is cut off "
    "by the bottom edge of the frame — the hand and forearm are the only part of a person "
    "in the picture. Natural indoor lighting, sharp focus on the hand and the object, "
    "realistic photo."
)

NEGATIVE = run_pickup.NEGATIVE + (
    ", forearm resting on the counter, wrist touching the countertop, "
    "arm lying flat on the counter, elbow on the counter, straight outstretched arm"
)

SPEC = PairSpec(
    stage1_prompt=lambda row: run_counter.VARIANTS["reach"].format(
        obj=row.object_description),
    stage1_negative=run_counter.NEGATIVE,
    stage1_file="before.png",
    stage2_prompt=lambda row: run_pickup.OPENING + run_pickup.GENERIC_GRIP
        + LIFT_CLAUSE + KEEP_CLAUSE,
    stage2_negative=NEGATIVE,
    stage2_file="after.png",
)

if __name__ == "__main__":
    main_cli(MODULE, require_grasp=False, description=__doc__)
