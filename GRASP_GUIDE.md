# Writing a functional-grasp spec for a new object

How to add an object to `GRASPS` in `scripts/run_grasp.py` and get an egocentric
image of it held in the grip a person would actually use.

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
fills `{grasp}`. Write only the grip and the object's pose — the camera framing,
the single bare forearm, and the hand exclusions already live in the three
`VARIANTS` and the shared `NEGATIVE`.

## Choosing the grip

Pick the grip used to do the thing the object is *for*, not the most convenient
way to lift it: a knife carried to a drawer is held differently from a knife
about to cut. Decide which task the object is mid-way through, then describe the
grip for that task.

| grip | when | tested on |
|---|---|---|
| power / wrap grip | handles, hafts, anything with torque | pizza peel, knife |
| hook grip | closed handles, loops, bails | mug |
| palmar drape | objects operated by the fingers while the palm rests on them | mouse |
| precision pinch | small or thin objects held between thumb and fingertips | *untested* |
| lateral / key pinch | flat objects gripped edge-on, thumb against index side | *untested* |
| tripod grip | pens, styluses, small tools | *untested* |

Adding an object with a genuinely new grip type teaches more than another
power-grip object, since the four existing entries were chosen to span grip types
*and* resting poses.

Some objects have no functional grip at all. An L-angle bracket is a part rather
than a tool, so nothing about holding it reads as "in use". There the task *is*
carrying it, so use the **transport grip**: the grip a person's hand falls into to
pick that object up and move it, determined by the object's shape, size and weight
alone. An L-bracket gets pinched by one flange or wrapped in the fingers because
that is what its shape affords. Expect the task-intent clause (slot 5) to have
nothing to say.

## The five slots

Every grasp description fills the same five slots. Work through them in order;
each exists because leaving it out produced a specific failure.

**1. Name the grip type.** *"a firm closed power grip"*, *"a hook grip"*, *"a
relaxed palmar grip"*. A named grip type anchors the whole hand configuration in
one phrase, and it is the single highest-leverage part of the spec: generic
*"holding it"* reads as merely picked up rather than in use.

**2. Spell out finger and thumb topology.** Which digits go where, with the thumb
called out separately — thumb placement is what distinguishes grips that share a
name. The peel and the knife are both power grips and the thumb is the only
difference: peel = *"thumb closed over the top of it"*, knife = *"thumb pressed
flat along the side of the handle just behind the bolster"*. Say what the
*non*-gripping fingers do too (*"the remaining fingers curled loosely beneath
it"*), or they get rendered doing something arbitrary.

**3. Give the contact location on the object.** *"The very end of the long wooden
handle"*, *"just behind the bolster"* — rather than only *"the handle"*. Where
along an object it is held separates a functional grip from an awkward one.

**4. State the object's axis, attitude and height.** Which way the working end
points (*"the blade points away from the hand toward the upper left of the
frame"*), its attitude (*"held level"*, *"angled slightly down"*, *"hangs upright
and level ... its open top facing the ceiling"*), and how far off the surface it
sits. Give clearance in units — *"raised about 30cm above the counter"* — because
vague clearance is the most reliably under-obeyed language in these prompts:
*"held level in mid-air well clear of the countertop"* produced an object hovering
only barely above it. Omitting the axis lets it drift, as the terse variant on the
peel did, rotating to a vertical straight-up-the-frame handle across two seeds.

**5. Close with task intent.** *"as if about to slide a pizza into an oven"*, *"as
if about to start slicing"*, *"as if being raised to drink from"*. This is what
makes the pose read as mid-task, and it pins height and angle better than a bare
*"lifted"* does.

The spec is done when all five slots are filled.

## Reading the results

`outputs/grasp_<slug>/review.html` holds three prompt variants × three seeds.
Read the variants as a diagnostic:

- **`explicit`** — full anatomical spec. Best of the three on the peel, and the
  default source for the place-down step.
- **`action`** — GoPro/"I am holding" phrasing, leaning on a photographic prior.
- **`terse`** — grip plus framing only. If `terse` holds, the spec is carrying its
  weight; if only `explicit` works, the grip needs the extra anatomy. If `terse`
  drifts while `explicit` holds, slot 4 is usually thin.

Judge grip correctness first, then anatomy (five fingers, plausible wrist), then
object fidelity. On the peel the object shape was the weak point rather than the
hand — the blade came out short and thick like a cutting board, and two seeds
rounded it into a paddle. That fix belongs in the *object description* (thin edge,
squared corners, handle longer than the blade), not the grasp description.

## Then the "before" frame

`scripts/run_place_down.py` inverts a grasp image — same object, set down on the
counter, hand gone. Add a label entry to its `OBJECTS` dict; its prompts are
object-agnostic and need no change. See the README for the `--ratio` asymmetry
between the two scripts.
