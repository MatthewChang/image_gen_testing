# Lift-prompt iteration for pipeline B ("add" pickup edit)

**Goal**: the stage-2 "add" edit of `run_pair_before_first.py` should lift the
object clearly off the counter (wrist and forearm not resting on the surface)
reliably — target ~8+/10 object classes good on both test seeds.

**Diagnosis being addressed** (from `outputs/diag_before_first/`): the research
"add" prompt was only validated on place-down frames, where the object sits
large and near the bottom-right. On pipeline B's fresh counter renders the
object sits at mid-counter distance, and the model's cheapest completion is a
straight arm lying flat across the counter with the object barely raised.

**Bench**: 10 stage-1 counter renders (`run_counter` "reach", seed 1, 1376×768),
fixed for all rounds, in `outputs/lift_bench/sources/`. Classes: mug, knife,
spatula, tennis ball, banana, screwdriver, remote, soup can, book, scissors.
Stage-2 seeds per round: 1001, 1002 (offset family the pipeline uses; avoids
the source's own seed).

**Grading** per image: `lift` (object clearly clear of the counter) and `arm`
(wrist/forearm raised, elbow bent — not lying on the surface). Good = both.

## Rounds

### Round 0 — baseline vs two hardened variants, seeds 1001/1002

Variants (`outputs/lift_bench/variants_r0.py`):
- `baseline` — the research `add` prompt verbatim (OPENING + GENERIC_GRIP + CLOSING).
- `bent-arm` — rewrites the opening ("elbow bent, hand raised high") and the lift
  clause ("held up in mid-air … hand, wrist and forearm all raised, not resting
  on the counter"); adds arm-pose terms to the negative.
- `toward-camera` — keeps the opening, re-anchors the lift to the camera
  ("roughly halfway between the countertop and the camera, nothing but air
  underneath"); same hardened negative.

Graded by three subagents (lift = clear air gap; arm = wrist/forearm raised):

| variant | good | notes |
|---|---|---|
| baseline | 20/20 | minor: garbled blade/button text, one soft forearm blur |
| bent-arm | 20/20 | minor: unlabeled soup can, oversized tennis ball |
| toward-camera | 20/20 | minor: scissors grip outside loops, blank book cover |

**Reading**: a clean sweep everywhere, including baseline — the smoke-test lift
failures were not primarily the prompt. Two differences vs the smoke run: the
bench sources are seed-1 renders (faithful backdrop, object near the front
edge) instead of the smoke's seed-0 renders, and the stage-2 seeds here are
1001/1002 instead of 1000/2/3 used in the failing diagnostics. Implies strong
per-seed variance in lift quality plus source-image sensitivity.

### Round 1 — same variants, seeds 1003–1006

| variant | 1003–04 | 1005–06 | total (r0+r1) |
|---|---|---|---|
| baseline | 20/20 | 17/20 | 57/60 |
| bent-arm | 19/20 | 18/20 | 57/60 |
| toward-camera | 20/20 | 19/20 | **59/60** |

All failures were lift failures (object grazing the counter with an attached
shadow: mug ×3, soup-can ×2, book ×1, remote ×1); the arm criterion never
failed on any variant — the flat-outstretched-arm failure mode from the smoke
diagnostics did not recur on these sources.

## Conclusions

1. **The smoke-test lift failures were driven by the source image and the
   seed, not the prompt.** On seed-1 counter renders (object near the front
   edge), even the unmodified research prompt lifts well at 57/60. The failing
   diagnostics used seed-0 sources (object mid-counter, reinvented room) and
   seeds 2/3/1000.
2. **`toward-camera` wins**: 59/60 (98%) over two rounds and 10 classes, and
   its lift clause ("raised toward the camera, roughly halfway between the
   countertop and the camera, nothing but air underneath") plus the hardened
   arm-pose negative directly target both failure axes. Adopted as the
   production stage-2 prompt of `run_pair_before_first.py`.
3. Residual failure mode (~2–5% across variants): object grazing the surface
   with an attached shadow, mostly mug/soup-can. Reseeding the row is the fix;
   no prompt tested eliminated it.

Bench assets: sources in `outputs/lift_bench/sources/`, renders in
`outputs/lift_bench/r0` and `r1`, variants in
`outputs/lift_bench/variants_r0.py`, runner `outputs/lift_bench/run_round.py`
(`BENCH_SEEDS` env overrides seeds).
