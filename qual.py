import numpy as np, matplotlib, sys, json
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "serif", "font.size": 8})
img = sys.argv[1] if len(sys.argv) > 1 else "rocket"
M = ["pe", "pe_ss", "ipe_gauss", "ipe_box", "ipe_hyb", "ipe_hyb_c"]
T = {"gt": "Ground truth", "pe": "NeRF-PE", "pe_ss": "PE + 16x SS", "ipe_gauss": "Gaussian IPE", "ipe_box": "Box-IPE", "ipe_hyb": "Lobe-IPE", "ipe_hyb_c": "Lobe-IPE + C"}
P = {m: np.load(f"preds/{img}_{m}.npz") for m in M}
rows = [("shift_2", "s=2 (seen)"), ("shift_3", "s=3 (unseen)"), ("shift_6", "s=6 (unseen)"), ("shift_16", "s=16 (seen)")]
# crop: same physical region (base-pixel window) for all scales
cx, cy, half = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5, float(sys.argv[3]) if len(sys.argv) > 3 else 0.5, 96
fig, ax = plt.subplots(len(rows), 7, figsize=(7.6, 1.25 * len(rows) + .2))
for i, (k, lab) in enumerate(rows):
    gt = P["pe"]["gt_" + k]; n = gt.shape[0]; s = int(k.split("_")[1])
    x0 = int((cx * 512 - half) / s); x1 = int((cx * 512 + half) / s); y0 = int((cy * 512 - half) / s); y1 = int((cy * 512 + half) / s)
    crop = lambda a: a[max(y0, 0):y1, max(x0, 0):x1]
    for j, m in enumerate(["gt"] + M):
        a = gt if m == "gt" else P[m][k]
        c = np.clip(crop(a), 0, 1)
        ax[i, j].imshow(c, interpolation="nearest"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
        if m != "gt":
            ps = -10 * np.log10(np.mean((c - crop(gt)) ** 2))
            ax[i, j].text(0.97, 0.03, f"{ps:.1f}", transform=ax[i, j].transAxes, ha="right", va="bottom", fontsize=7,
                          color="white", bbox=dict(facecolor="black", alpha=.55, pad=1, lw=0))
        if i == 0: ax[i, j].set_title(T[m], fontsize=8)
        if j == 0: ax[i, j].set_ylabel(lab, fontsize=8)
plt.tight_layout(pad=0.2); plt.savefig(f"figs/fig_qual_{img}.png", dpi=300)
