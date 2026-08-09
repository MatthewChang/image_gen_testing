# Writing a functional-grasp spec for a new object

How to add an object to `GRASPS` in `scripts/run_grasp.py` and get an egocentric
image of it held in the grip a person would actually use.

Everything here is derived from the four worked objects in that dict
(pizza peel, kitchen knife, mug, computer mouse) and from what the sweeps
actually produced. Where a claim is only tested on those four, it says so.

## What a spec is

Each entry is `slug: (object_description, grasp_description)`:

```python
"mug": (
    "a plain ceramic mug with a curved side handle",
    "The index and middle fingers are hooked through the mug's handle in a hook grip with "
    "the thumb resting on top of the handle, the remaining fingers curled loosely beneath "
    "it. The mug hangs upright and level from the handle, its open top facing the ceiling, "
    "lifted clear of the counter as if being raised to drink from"
),
```

The object description fills `{obj}` in the framing prompt; the grasp description
fills `{grasp}`. The framing (head-mounted camera, one bare forearm entering from
the bottom of the frame, nothing else of the person) is already handled by the
three `VARIANTS` and the shared `NEGATIVE` — **do not restate framing or hand
exclusions in the spec.** Write only the grip and the object's pose.

## The five slots

Every grasp description in the dict fills the same five slots. Work through them
in order; each one exists because leaving it out produced a specific failure.

**1. Name the grip type.** *"a firm closed power grip"*, *"a hook grip"*, *"a
relaxed palmar grip"*. A named grip type anchors the whole hand configuration in
one phrase. Generic *"holding it"* reads as *merely picked up* rather than *in
use* — this is the single highest-leverage part of the spec.

**2. Spell out finger and thumb topology.** Which digits go where, separately for
the thumb, because thumb placement is what distinguishes grips that share a name.
The peel and the knife are both power grips, and the thumb is the only difference:
peel = *"thumb closed over the top of it"*, knife = *"thumb pressed flat along the
side of the handle just behind the bolster"*. Also say what the *non*-gripping
fingers do (*"the remaining fingers curled loosely beneath it"*), or they get
rendered doing something arbitrary.

**3. Give the contact location on the object.** Not just *"the handle"* but *"the
very end of the long wooden handle"*, *"just behind the bolster"*. Where along an
object it's held is what separates a functional grip from an awkward one.

**4. State the object's axis and orientation in frame.** Which way the working
end points (*"the blade points away from the hand toward the upper left of the
frame"*), and its attitude (*"held level"*, *"angled slightly down"*, *"hangs
upright and level ... its open top facing the ceiling"*). Without this the long
axis drifts — the terse variant on the peel drifted to a vertical
straight-up-the-frame handle across two seeds.

**5. Close with task intent.** *"as if about to slide a pizza into an oven"*, *"as
if about to start slicing"*, *"as if being raised to drink from"*. This is what
makes the pose read as mid-task. It also implicitly fixes height and angle better
than a bare *"lifted"* does.

## Height off the surface

Say it, and say it in units. *"held level in mid-air well clear of the
countertop"* was only half-obeyed — the object hovers, but often barely. If
clearance matters, write *"raised about 30cm above the counter"*. Vague
clearance language is the most reliably under-obeyed part of these prompts.

## Pick the grip a person actually uses

The point is the *functional* grip — the one used to do the thing the object is
for, not the most convenient way to lift it. A knife carried to a drawer is held
differently from a knife about to cut. Decide what task the object is mid-way
through, then describe the grip for that task.

Useful grip vocabulary, with the tested example alongside:

| grip | when | tested on |
|---|---|---|
| power / wrap grip | handles, hafts, anything with torque | pizza peel, knife |
| hook grip | closed handles, loops, bails | mug |
| palmar drape | objects operated by the fingers while the palm rests on them | mouse |
| precision pinch | small or thin objects held between thumb and fingertips | *untested* |
| lateral / key pinch | flat objects gripped edge-on, thumb against index side | *untested* |
| tripod grip | pens, styluses, small tools | *untested* |

The first four `GRASPS` entries were deliberately chosen to span different grip
types *and* different resting poses, since that is what stresses the downstream
placement prompt in `run_place_down.py`. Adding an object with a genuinely new
grip type is more informative than another power-grip object.

## Objects with no functional grasp

Not every object has one. An L-angle bracket is a part, not a tool — there is no
grip that means "in use". For these, the honest spec is a precision pinch that
presents the object clearly, and you should expect the "as if..." intent clause
to have nothing useful to say. Don't invent a task that doesn't exist; a
transport grip described accurately beats a fictional functional one.

## Checklist

Before running, confirm the spec:

- [ ] names a grip type
- [ ] places the thumb explicitly, separately from the fingers
- [ ] says what the non-gripping fingers do
- [ ] gives the contact location along the object, not just the part
- [ ] states which way the working end points, and the object's attitude
- [ ] gives height above the surface, in units if it matters
- [ ] ends with an "as if ..." task-intent clause
- [ ] mentions no hands-in-frame constraints, no camera framing, no lighting
      (all of that lives in `VARIANTS` / `NEGATIVE`)

## Then run it

```bash
python scripts/run_grasp.py --object <slug> --ratio 16:9
```

Three prompt variants × three seeds; review sheet at
`outputs/grasp_<slug>/review.html`. Read the variants as a diagnostic:

- **`explicit`** — full anatomical spec. Was the best of the three on the peel,
  and the default source for the place-down step.
- **`action`** — GoPro/"I am holding" phrasing, leaning on a photographic prior.
- **`terse`** — grip plus framing only. If `terse` is fine, the spec is doing its
  job; if only `explicit` works, the grip needs the extra anatomy. If `terse`
  drifts while `explicit` holds, that's usually slot 4 (axis/orientation) missing.

Judge grip correctness first, then anatomy (five fingers, plausible wrist), then
object fidelity. On the peel, object shape was the weak point rather than the
hand — the blade came out short and thick like a cutting board, and two seeds
rounded it into a paddle. If that happens, the fix belongs in the *object
description* (thin edge, squared corners, handle longer than the blade), not in
the grasp description.

## Then the "before" frame

`scripts/run_place_down.py` inverts a grasp image — same object, set down on the
counter, hand gone. Add a label entry to its `OBJECTS` dict; its prompts are
object-agnostic and need no change. See the README for the `--ratio` asymmetry
between the two scripts.
