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
   - `seed`: 1, 2, ... within each class (never 0 — seed 0 at the production
     dims tends to reinvent the room background)

4. Write the manifest, then validate:

       python scripts/pairs_lib.py <manifest> --require-grasp

   Fix and re-run until it prints `N row(s) OK`.

Done when the validator passes and every grasp description fills all five
slots — reread each against the guide's slot list before reporting the
manifest path and any interpretation choices from step 1.
