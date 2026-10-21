import json, numpy as np, collections, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "mathtext.fontset": "cm"})

R = [json.loads(l) for l in open("results.jsonl")]
NAMES = {"pe_single": "NeRF-PE (single-scale)", "pe": "NeRF-PE (multiscale)", "pe_ss": "PE + 16x SS",
         "ipe_gauss": "Gaussian IPE (mip-NeRF)", "ipe_box": "Box-IPE (ours)", "ipe_hyb": "Lobe-IPE (ours)",
         "ipe_gauss_c": "Gaussian IPE + C", "ipe_hyb_c": "Lobe-IPE + C (ours)"}
ORDER = ["pe_single", "pe", "pe_ss", "ipe_gauss", "ipe_box", "ipe_hyb", "ipe_gauss_c", "ipe_hyb_c"]
COL = {"pe_single": "#999999", "pe": "#7f7f7f", "pe_ss": "#8e44ad", "ipe_gauss": "#1f6fb4", "ipe_box": "#c0392b", "ipe_hyb": "#27ae60", "ipe_gauss_c": "#5dade2", "ipe_hyb_c": "#e67e22"}
SH = [2, 3, 4, 6, 8, 12, 16]; SEEN = [2, 4, 8, 16]; UNSEEN = [3, 6, 12]
out = {}


def agg(rs, key, metric):
    return np.array([r["eval"][key][metric] for r in rs])


def table(tag, methods, L=10):
    T = {}
    for m in methods:
        rs = [r for r in R if r["tag"] == tag and r["method"] == m and r["L"] == L]
        if not rs: continue
        row = {"n": len(rs)}
        for s in SH:
            row[f"psnr_{s}"] = float(agg(rs, f"shift_{s}", "psnr").mean())
            row[f"ssim_{s}"] = float(agg(rs, f"shift_{s}", "ssim").mean())
        row["psnr_seen"] = float(np.mean([row[f"psnr_{s}"] for s in SEEN if f"shift_{s}" in rs[0]["eval"]]))
        row["psnr_unseen"] = float(np.mean([row[f"psnr_{s}"] for s in UNSEEN]))
        row["psnr_all"] = float(np.mean([row[f"psnr_{s}"] for s in SH]))
        row["ssim_all"] = float(np.mean([row[f"ssim_{s}"] for s in SH]))
        row["ongrid_psnr"] = float(np.mean([agg(rs, k, "psnr").mean() for k in rs[0]["eval"] if k.startswith("ongrid")]))
        row["train_min"] = float(np.mean([r["train_time_s"] for r in rs]) / 60)
        row["sec_per_mp"] = float(np.mean([r["eval"]["shift_2"]["sec_per_mp"] for r in rs]))
        # per-run std of the scale-averaged PSNR
        row["psnr_all_std"] = float(np.std([np.mean([r["eval"][f"shift_{s}"]["psnr"] for s in SH]) for r in rs]))
        T[m] = row
    return T


out["main"] = table("main", ORDER)
out["sparse"] = table("sparse", ORDER)


def paired(tag, a, b, scales):
    da, db = {}, {}
    for r in R:
        if r["tag"] != tag or r["L"] != 10: continue
        for s in scales:
            k = (r["image"], r["seed"], s)
            if r["method"] == a: da[k] = r["eval"][f"shift_{s}"]["psnr"]
            if r["method"] == b: db[k] = r["eval"][f"shift_{s}"]["psnr"]
    ks = sorted(set(da) & set(db))
    d = np.array([da[k] - db[k] for k in ks])
    if len(d) < 3: return None
    p = wilcoxon(d).pvalue
    return {"n": len(d), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()), "p": float(p)}


out["tests"] = {}
for tag in ["main", "sparse"]:
    for (a, b) in [("ipe_box", "ipe_gauss"), ("ipe_hyb", "ipe_gauss"), ("ipe_hyb", "ipe_box"), ("ipe_box", "pe"), ("ipe_hyb_c", "ipe_hyb"),
                   ("ipe_hyb_c", "ipe_gauss_c"), ("ipe_gauss_c", "ipe_gauss"), ("ipe_hyb_c", "pe_ss"), ("ipe_hyb_c", "ipe_gauss")]:
        for nm, sc in [("all", SH), ("seen", SEEN), ("unseen", UNSEEN)]:
            out["tests"][f"{tag}:{a}-{b}:{nm}"] = paired(tag, a, b, sc)

# L sweep (include L=10 from main seed 0 on the same images)
Ls = {}
for m in ["pe", "ipe_gauss", "ipe_box"]:
    for L in [6, 8, 10, 12, 16]:
        rs = [r for r in R if r["method"] == m and r["L"] == L and r["image"] in ("astronaut", "coffee") and r["seed"] == 0
              and r["tag"] in ("Lsweep", "main")]
        if rs:
            Ls[f"{m}_{L}"] = {"psnr_all": float(np.mean([np.mean([r["eval"][f"shift_{s}"]["psnr"] for s in SH]) for r in rs])),
                              "psnr_unseen": float(np.mean([np.mean([r["eval"][f"shift_{s}"]["psnr"] for s in UNSEEN]) for r in rs])),
                              "n": len(rs)}
out["Lsweep"] = Ls

# per-image (main, both seeds) scale-averaged PSNR
per = collections.defaultdict(dict)
for m in ORDER:
    for img in ["astronaut", "chelsea", "coffee", "rocket", "ihc", "retina"]:
        rs = [r for r in R if r["tag"] == "main" and r["method"] == m and r["image"] == img]
        if rs: per[img][m] = float(np.mean([np.mean([r["eval"][f"shift_{s}"]["psnr"] for s in SH]) for r in rs]))
out["per_image"] = per
out["collapsed"] = {}
for r in R:
    k = f'{r["tag"]}:{r["method"]}'
    out["collapsed"].setdefault(k, 0)
    if np.mean([v["psnr"] for kk, v in r["eval"].items() if kk.startswith("shift")]) < 10: out["collapsed"][k] += 1
json.dump(out, open("stats.json", "w"), indent=1)

# ---- figure: PSNR vs scale
for tag, fn in [("main", "fig_scale_main.png"), ("sparse", "fig_scale_sparse.png")]:
    T = out[tag]
    if not T: continue
    fig, ax = plt.subplots(1, 2, figsize=(7, 3.0))
    for m in ORDER:
        if m not in T or T[m]["psnr_all"] < 15 and tag == "sparse": continue
        ax[0].plot(SH, [T[m][f"psnr_{s}"] for s in SH], "o-", color=COL[m], ms=3.5, lw=1.4, label=NAMES[m])
        ax[1].plot(SH, [T[m][f"ssim_{s}"] for s in SH], "o-", color=COL[m], ms=3.5, lw=1.4)
    tr = SEEN if tag == "main" else [2, 16]
    for a in ax:
        a.set_xscale("log", base=2); a.set_xticks(SH); a.set_xticklabels([str(s) for s in SH])
        a.set_xlabel("downsampling factor s (shifted grid)")
        for s in SH:
            if s not in tr: a.axvspan(s / 1.12, s * 1.12, color="#f3e6c8", lw=0, zorder=0)
    ax[0].set_ylabel("PSNR (dB)"); ax[1].set_ylabel("SSIM")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    plt.tight_layout(rect=(0, 0.12, 1, 1)); plt.savefig("figs/" + fn, dpi=300); plt.close()

# ---- figure: L sweep
if Ls:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for m in ["pe", "ipe_gauss", "ipe_box"]:
        xs = [L for L in [6, 8, 10, 12, 16] if f"{m}_{L}" in Ls]
        ax.plot(xs, [Ls[f"{m}_{L}"]["psnr_all"] for L in xs], "o-", color=COL[m], ms=3.5, label=NAMES[m])
    ax.set_xlabel("number of frequency levels L"); ax.set_ylabel("mean shifted PSNR (dB)")
    ax.legend(fontsize=6.5, frameon=False); plt.tight_layout(); plt.savefig("figs/fig_L.png", dpi=300); plt.close()
print(json.dumps({k: {m: round(v["psnr_all"], 2) for m, v in out[k].items()} for k in ["main", "sparse"]}, indent=0))
