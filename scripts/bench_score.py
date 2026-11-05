"""Score robot-swap outputs and build the review viewer.

    pixi run -e bench python scripts/bench_score.py

Needs masks from bench_segment.py. Per output (resized to the source crop):
  limb_iou    IoU of hand∪arm silhouettes, source vs output   (pose/placement)
  hand_iou    IoU of the hand masks alone                       (hand pose)
  hand_shift  hand-centroid offset, % of image diagonal         (position)
  bg_psnr     PSNR outside the dilated union of both limb masks (background kept)
  bg_lpips    LPIPS(alex) on the same region, limb area greyed  (lower = better)
Writes outputs/bench_robot_swap/scores.json and the viewer at
agent_scratch/viewers/<date>_bench_robot_swap/index.html.
"""

from __future__ import annotations

import datetime
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "outputs/bench_robot_swap"
MASKS = BENCH / "_masks"
OVER = BENCH / "_overlays"
REFS = ROOT / "inputs/robot_swap/refs"
VIEWERS = ROOT / "agent_scratch/viewers"
TAG = "bench_robot_swap"
DILATE = 31  # px margin around the limbs excluded from the background metrics
DEV = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_INFO = {
    "qwen2511": ("Qwen-Image-Edit-2511", "baseline · 20B · 40 steps · Apache-2.0"),
    "firered11": ("FireRed-Image-Edit-1.1", "20B (Qwen-Image base) · 40 steps · Apache-2.0"),
    "qwen21turbo": ("Qwen-Image-2.1-Turbo", "7B · 8 steps · 2752×1536 · research licence"),
    "flux3klein": ("FLUX 3 Image Klein (base)", "5.6B + 9B upsampler · 50 steps · Kommunity"),
    "flux2klein4b": ("FLUX.2 [klein] 4B", "4B distilled · 4 steps · Apache-2.0"),
}
METRICS = [("limb_iou", "limb IoU", "↑"), ("hand_iou", "hand IoU", "↑"), ("hand_shift", "hand shift %", "↓"),
           ("bg_psnr", "bg PSNR", "↑"), ("bg_lpips", "bg LPIPS", "↓"), ("seconds", "s/edit", "↓")]


def load_mask(rel: str, part: str) -> np.ndarray:
    return np.array(Image.open(MASKS / f"{rel}__{part}.png")) > 127


def iou(a, b):
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else float("nan")


def centroid(m):
    ys, xs = np.nonzero(m)
    return (xs.mean(), ys.mean()) if len(xs) else None


def dilate(m: np.ndarray, r: int) -> np.ndarray:
    t = torch.from_numpy(m.astype(np.float32)).to(DEV)[None, None]
    return (F.max_pool2d(t, 2 * r + 1, stride=1, padding=r)[0, 0] > 0).cpu().numpy()


def overlay(img: np.ndarray, src_limb: np.ndarray, out_limb: np.ndarray) -> Image.Image:
    """Output tinted green where its limb is; source limb outline in red."""
    o = img.astype(np.float32).copy()
    o[out_limb] = 0.55 * o[out_limb] + 0.45 * np.array([40, 220, 90])
    edge = dilate(src_limb, 2) & ~src_limb
    o[edge] = [255, 40, 40]
    return Image.fromarray(o.clip(0, 255).astype(np.uint8))


def main() -> None:
    import lpips

    lp = lpips.LPIPS(net="alex", verbose=False).eval().to(DEV)
    sources = {p.stem: p for p in sorted((BENCH / "_inputs").glob("*.png"))}
    models = [m for m in MODEL_INFO if (BENCH / m).is_dir()]
    rows = []
    for m in models:
        (OVER / m).mkdir(parents=True, exist_ok=True)
        for p in sorted((BENCH / m).glob("*.png")):
            frame, cond, seed = p.stem.split("__")
            meta = json.loads(p.with_suffix(".json").read_text())
            src_rel, out_rel = f"_inputs/{frame}", f"{m}/{p.stem}"
            if not (MASKS / f"{out_rel}__limb.png").exists():
                print(f"no mask yet: {out_rel}")
                continue
            sh, sl = load_mask(src_rel, "hand"), load_mask(src_rel, "limb")
            oh, ol = load_mask(out_rel, "hand"), load_mask(out_rel, "limb")
            src = np.array(Image.open(sources[frame]).convert("RGB"))
            out = np.array(Image.open(p).convert("RGB").resize(src.shape[1::-1], Image.LANCZOS))

            bg = ~dilate(sl | ol, DILATE)
            mse = float(((src[bg].astype(np.float32) - out[bg]) ** 2).mean())
            psnr = 10 * math.log10(255 ** 2 / mse) if mse > 0 else 99.0
            to_t = lambda a: torch.from_numpy(np.where(bg[..., None], a, 128).astype(np.float32)
                                              / 127.5 - 1).permute(2, 0, 1)[None]
            small = lambda t: F.interpolate(t.to(DEV), size=(288, 512), mode="area")
            with torch.no_grad():
                lpv = float(lp(small(to_t(src)), small(to_t(out))))
            cs, co = centroid(sh), centroid(oh)
            diag = math.hypot(*src.shape[:2])
            shift = math.hypot(cs[0] - co[0], cs[1] - co[1]) / diag * 100 if cs and co else float("nan")

            ov = OVER / m / f"{p.stem}.jpg"
            if not ov.exists():
                overlay(out, sl, ol).save(ov, quality=88)
            rows.append({"model": m, "frame": frame, "cond": cond, "seed": int(seed[1:]),
                         "file": f"{m}/{p.name}", "overlay": f"{m}/{ov.name}",
                         "limb_iou": iou(sl, ol), "hand_iou": iou(sh, oh), "hand_shift": shift,
                         "bg_psnr": psnr, "bg_lpips": lpv, "seconds": meta["seconds"],
                         "size": meta["size"], "hand_found": bool(oh.any()),
                         "upsampled_prompt": meta.get("upsampled_prompt")})
            print(f"{out_rel}: limb {rows[-1]['limb_iou']:.2f} hand {rows[-1]['hand_iou']:.2f} "
                  f"psnr {psnr:.1f} lpips {lpv:.3f}", flush=True)

    # Source overlays (outline only) so the viewer can show the target silhouette.
    (OVER / "_inputs").mkdir(parents=True, exist_ok=True)
    for f, p in sources.items():
        ov = OVER / "_inputs" / f"{f}.jpg"
        if not ov.exists():
            sl = load_mask(f"_inputs/{f}", "limb")
            overlay(np.array(Image.open(p).convert("RGB")), sl, np.zeros_like(sl)).save(ov, quality=88)

    summary = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for k, *_ in METRICS:
            if not math.isnan(r[k]):
                summary[(r["model"], r["cond"])][k].append(r[k])
    table = [{"model": m, "cond": c, "n": len(v["limb_iou"]),
              **{k: float(np.mean(v[k])) if v[k] else None for k, *_ in METRICS}}
             for (m, c), v in sorted(summary.items(), key=lambda kv: (models.index(kv[0][0]), kv[0][1]))]
    (BENCH / "scores.json").write_text(json.dumps({"rows": rows, "summary": table}, indent=2))
    build_viewer(rows, table, models, sorted(sources))


def build_viewer(rows, table, models, frames) -> None:
    existing = sorted(VIEWERS.glob(f"*_{TAG}")) if VIEWERS.exists() else []
    vdir = existing[0] if existing else VIEWERS / f"{datetime.date.today():%Y-%m-%d}_{TAG}"
    vdir.mkdir(parents=True, exist_ok=True)
    links = {"out": BENCH, "ov": OVER, "refs": REFS}
    for name, target in links.items():
        link = vdir / name
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(target, link)
    for r in rows:
        for rel in (f"out/{r['file']}", f"ov/{r['overlay']}"):
            assert (vdir / rel).exists(), rel
    for f in frames:
        assert (vdir / f"out/_inputs/{f}.png").exists() and (vdir / f"ov/_inputs/{f}.jpg").exists(), f
    for f in ("sharpa_hand.png", "tianji_arm.png"):
        assert (vdir / "refs" / f).exists(), f

    data = {"rows": rows, "summary": table, "frames": frames, "models": models,
            "info": {m: MODEL_INFO[m] for m in models}, "metrics": METRICS,
            "prompts": json.loads(json.dumps(__import__("bench_robot_swap").PROMPTS))}
    page = HTML.replace("__DATA__", json.dumps(data).replace("</", "<\\/")) \
               .replace("__GEN__", str(Path(__file__).resolve()))
    (vdir / "index.html").write_text(page)
    print(f"viewer: {vdir / 'index.html'}")


HTML = r"""<!doctype html>
<html><head><meta charset="utf-8">
<meta name="generator" content="__GEN__">
<title>Robot-swap edit benchmark</title>
<style>
:root{color-scheme:dark}
body{font:14px/1.5 system-ui,sans-serif;margin:0;padding:1.2rem 1.6rem;background:#121418;color:#e6e6e8}
h1{font-size:1.35rem;margin:0 0 .2rem} h2{font-size:1.05rem;margin:1.4rem 0 .5rem;color:#8fd3ff}
.meta{color:#9aa0a8;font-size:.85rem;margin:0 0 .8rem}
.prompt{background:#1c1f25;border-left:3px solid #4a5058;padding:.4rem .7rem;margin:.3rem 0;font-size:.82rem;border-radius:3px}
.refs{display:flex;gap:.8rem;align-items:flex-start}.refs figure{margin:0;width:180px}
.refs img{width:100%;border-radius:4px;background:#fff}
table{border-collapse:collapse;font-size:.85rem;margin:.4rem 0}
th,td{padding:.25rem .6rem;border-bottom:1px solid #2a2e35}th{text-align:left;color:#9aa0a8;font-weight:500}
td.n{text-align:right;font-variant-numeric:tabular-nums}td.best{color:#7dff9b;font-weight:600}
.controls{position:sticky;top:0;background:#121418;padding:.5rem 0;z-index:5;display:flex;gap:1.2rem;flex-wrap:wrap;border-bottom:1px solid #2a2e35}
.controls label{cursor:pointer;margin-right:.5rem}
.grid{display:grid;gap:.5rem;margin-top:.6rem}
.cell{min-width:0}.cell img{width:100%;display:block;border-radius:3px;cursor:zoom-in;background:#000}
.cap{font:11px/1.35 ui-monospace,monospace;color:#9aa0a8;margin-top:.15rem}
.cap b{color:#e6e6e8}.bad{color:#ff8a8a}
.hdr{font-size:.8rem;color:#8fd3ff;font-weight:600}.hdr small{display:block;color:#9aa0a8;font-weight:400}
.rowlab{font:12px ui-monospace,monospace;color:#e6e6e8;align-self:center}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.92);display:none;align-items:center;justify-content:center;z-index:9;flex-direction:column}
#lb img{max-width:96vw;max-height:88vh}#lb div{color:#ccc;font:12px ui-monospace,monospace;margin-top:.4rem;max-width:90vw}
</style></head><body>
<h1>Robot-swap edit benchmark — human right hand → Sharpa Wave hand on Tianji Marvin arm</h1>
<p class="meta">9 egocentric frames (vid_test cond1–3) × {text-only, with references} × 3 seeds × each model.
Scores compare each output (resized to the source) with its source: silhouettes from SAM 3 text prompts.
Overlay view: green = output limb mask, red outline = source limb.</p>
<div id="prompts"></div>
<div class="refs" id="refs"></div>
<h2>Summary (mean over frames × seeds)</h2>
<div id="summary"></div>
<h2>Side by side</h2>
<div class="controls" id="controls"></div>
<div class="grid" id="grid"></div>
<div id="lb" onclick="this.style.display='none'"><img><div></div></div>
<script>
const D = __DATA__;
const S = Object.assign({cond:"refs", seed:"0", view:"img"}, Object.fromEntries(new URLSearchParams(location.hash.slice(1))));
const fmt = (k,v) => v==null||Number.isNaN(v) ? "–" : (k==="bg_psnr"||k==="seconds"||k==="hand_shift") ? v.toFixed(1) : v.toFixed(3);
function save(){ location.hash = new URLSearchParams(S).toString(); render(); }

document.getElementById("prompts").innerHTML = Object.entries(D.prompts)
  .map(([k,v]) => `<div class="prompt"><b>${k}</b>: ${v}</div>`).join("");
document.getElementById("refs").innerHTML =
  `<figure><img src="refs/sharpa_hand.png"><div class="cap">ref 2: Sharpa Wave hand</div></figure>
   <figure><img src="refs/tianji_arm.png"><div class="cap">ref 3: Tianji Marvin arm</div></figure>`;

function summary(){
  const ms = D.metrics, best = {};
  for (const [k,,dir] of ms){ const vs = D.summary.map(r=>r[k]).filter(v=>v!=null);
    best[k] = dir==="↑" ? Math.max(...vs) : Math.min(...vs); }
  let h = `<table><tr><th>model</th><th>cond</th><th>n</th>${ms.map(([,l,d])=>`<th>${l} ${d}</th>`).join("")}</tr>`;
  for (const r of D.summary){
    h += `<tr><td>${D.info[r.model][0]}</td><td>${r.cond}</td><td class="n">${r.n}</td>` +
      ms.map(([k])=>`<td class="n ${r[k]===best[k]?"best":""}">${fmt(k,r[k])}</td>`).join("") + "</tr>";
  }
  document.getElementById("summary").innerHTML = h + "</table>";
}

function controls(){
  const radio = (name, opts) => `<span>${name}: ` + opts.map(([v,l]) =>
    `<label><input type="radio" name="${name}" value="${v}" ${S[name]===v?"checked":""}> ${l}</label>`).join("") + "</span>";
  const el = document.getElementById("controls");
  el.innerHTML = radio("cond", [["text","text-only"],["refs","with refs"]]) +
    radio("seed", [["0","0"],["1","1"],["2","2"]]) + radio("view", [["img","image"],["ov","mask overlay"]]);
  el.querySelectorAll("input").forEach(i => i.onchange = () => { S[i.name] = i.value; save(); });
}

function render(){
  const g = document.getElementById("grid");
  g.style.gridTemplateColumns = `90px repeat(${D.models.length + 1}, minmax(0,1fr))`;
  let h = `<div></div><div class="hdr">source</div>` +
    D.models.map(m => `<div class="hdr">${D.info[m][0]}<small>${D.info[m][1]}</small></div>`).join("");
  const idx = {}; for (const r of D.rows) idx[[r.model,r.frame,r.cond,r.seed]] = r;
  for (const f of D.frames){
    const src = S.view==="ov" ? `ov/_inputs/${f}.jpg` : `out/_inputs/${f}.png`;
    h += `<div class="rowlab">${f}</div><div class="cell"><img src="${src}" data-cap="${f} source"></div>`;
    for (const m of D.models){
      const r = idx[[m,f,S.cond,+S.seed]];
      if (!r){ h += `<div class="cell cap">not run</div>`; continue; }
      const src = S.view==="ov" ? `ov/${r.overlay}` : `out/${r.file}`;
      const cap = `limb <b>${fmt("limb_iou",r.limb_iou)}</b> hand <b>${fmt("hand_iou",r.hand_iou)}</b> ` +
        `shift ${fmt("hand_shift",r.hand_shift)}%<br>bg ${fmt("bg_psnr",r.bg_psnr)}dB lpips ${fmt("bg_lpips",r.bg_lpips)}` +
        (r.hand_found ? "" : ` <span class="bad">no hand found</span>`);
      const full = `${D.info[m][0]} · ${f} · ${S.cond} · seed ${S.seed}` + (r.upsampled_prompt ? ` — upsampled: ${r.upsampled_prompt}` : "");
      h += `<div class="cell"><img loading="lazy" src="${src}" data-cap="${full.replace(/"/g,"&quot;")}"><div class="cap">${cap}</div></div>`;
    }
  }
  g.innerHTML = h;
  g.querySelectorAll("img").forEach(i => i.onclick = () => {
    const lb = document.getElementById("lb"); lb.querySelector("img").src = i.src;
    lb.querySelector("div").textContent = i.dataset.cap; lb.style.display = "flex"; });
}
summary(); controls(); render();
window.onhashchange = () => { Object.assign(S, Object.fromEntries(new URLSearchParams(location.hash.slice(1)))); controls(); render(); };
</script></body></html>
"""

if __name__ == "__main__":
    main()
