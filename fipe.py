"""Footprint-integrated positional encodings for multiscale coordinate networks.

Protocol (2D analogue of mip-NeRF's multiscale Blender benchmark)
  * a 512x512 base image B defines the continuous signal; every observation is a
    box-filtered pixel: the mean of B over an s x s block of base pixels.
  * TRAIN: on-grid pyramids at factors s in {2,4,8,16} (256^2 .. 32^2 images), grid offset 0.
  * TEST (held out): pixel grids shifted by floor(s/2) base pixels (new pixel centres
    and footprints never observed), at training factors AND unseen factors {3,6,12}.
    This mimics rendering new camera poses / zoom levels.

Methods
  pe_single : NeRF PE, trained on the finest scale only, point-sampled at pixel centres
  pe        : PE, multiscale training, point-sampled at pixel centres
  pe_ss     : PE + brute-force supersampling (K=4 jittered points/pixel train, 4x4 test)
  ipe_gauss : mip-NeRF IPE, Gaussian footprint (var = w^2/12)
  ipe_box   : exact box-footprint IPE (sinc attenuation)              [proposed]
  ipe_hyb   : box IPE truncated to the non-negative main lobe         [proposed variant]
"""
import json, math, os, time, argparse
import numpy as np
import torch, torch.nn as nn
from skimage import data, transform
from skimage.metrics import structural_similarity

torch.set_num_threads(int(os.environ.get("NT", "2")))
BASE = 512
BW = 2.0 / BASE  # width of a base pixel in [-1,1] coordinates

IMAGES = {
    "astronaut": lambda: data.astronaut(),
    "chelsea": lambda: data.chelsea(),
    "coffee": lambda: data.coffee(),
    "rocket": lambda: data.rocket(),
    "ihc": lambda: data.immunohistochemistry(),
    "retina": lambda: data.retina(),
}


def load_image(name):
    im = IMAGES[name]().astype(np.float32) / 255.0
    h, w = im.shape[:2]
    s = min(h, w)
    im = im[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    if s != BASE:
        im = transform.resize(im, (BASE, BASE), anti_aliasing=True, order=3)
    return np.clip(im, 0, 1).astype(np.float32)


def observe(B, s, o):
    """box-filtered observation at factor s with grid offset o (base px). returns img, centres, width."""
    n = (BASE - o) // s
    crop = B[o:o + n * s, o:o + n * s]
    img = crop.reshape(n, s, n, s, 3).mean(axis=(1, 3))
    c = -1 + (o + (np.arange(n) + 0.5) * s) * BW
    yy, xx = np.meshgrid(c, c, indexing="ij")
    X = np.stack([xx, yy], -1).reshape(-1, 2).astype(np.float32)
    return img.astype(np.float32), X, s * BW


def freqs(L):
    return torch.tensor([(2.0 ** l) * math.pi for l in range(L)], dtype=torch.float32)


def attenuation(z, mode):
    """z = omega * w / 2 (half-width phase)."""
    if mode == "ipe_gauss":
        return torch.exp(-z ** 2 / 6.0)                      # exp(-1/2 omega^2 w^2/12)
    sinc = torch.where(z > 1e-6, torch.sin(z) / z.clamp_min(1e-6), torch.ones_like(z))
    if mode == "ipe_box":
        return sinc
    if mode == "ipe_hyb":
        return torch.where(z < math.pi, sinc, torch.zeros_like(z))
    raise ValueError(mode)


def encode(x, w, L, mode):
    """x: (B,2) centres, w: (B,1) footprint widths -> (B, 4L)."""
    om = freqs(L)
    arg = x[:, :, None] * om                                  # (B,2,L)
    if mode == "pe":
        f = torch.cat([torch.sin(arg), torch.cos(arg)], -1)
    else:
        att = attenuation(om * w[:, :, None] / 2.0, mode).expand_as(arg)
        f = torch.cat([torch.sin(arg) * att, torch.cos(arg) * att], -1)
    return f.reshape(x.shape[0], -1)


class MLP(nn.Module):
    def __init__(self, d_in, width=256, depth=4):
        super().__init__()
        layers, d = [], d_in
        for _ in range(depth):
            layers += [nn.Linear(d, width), nn.ReLU()]
            d = width
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)

    def forward(self, f):
        return torch.sigmoid(self.net(f))


def build_train(B, scales):
    X, W, Y, A = [], [], [], []
    for s in scales:
        img, c, w = observe(B, s, 0)
        X.append(c); W.append(np.full((len(c), 1), w, np.float32)); Y.append(img.reshape(-1, 3))
        A.append(np.full((len(c),), float(s * s), np.float32))   # mip-NeRF area loss
    cat = lambda v: torch.from_numpy(np.concatenate(v))
    return cat(X), cat(W), cat(Y), cat(A)


def predict(model, x, w, L, method, K_test=4, chunk=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(x), chunk):
            xb, wb = x[i:i + chunk], w[i:i + chunk]
            if method == "pe_ss":
                g = (torch.arange(K_test, dtype=torch.float32) + 0.5) / K_test - 0.5
                gy, gx = torch.meshgrid(g, g, indexing="ij")
                off = torch.stack([gx.reshape(-1), gy.reshape(-1)], -1)
                pts = xb[:, None, :] + off[None] * wb[:, None, :]
                out.append(model(encode(pts.reshape(-1, 2), None, L, "pe")).reshape(len(xb), -1, 3).mean(1))
            else:
                mode = "pe" if method in ("pe", "pe_single") else method
                out.append(model(encode(xb, wb, L, mode)))
    return torch.cat(out)


def psnr(a, b):
    return float(-10 * np.log10(np.mean((a - b) ** 2)))


def run(img_name, method, seed=0, L=10, steps=2500, batch=4096, lr=1e-3,
        train_scales=(2, 4, 8, 16), test_scales=(2, 3, 4, 6, 8, 12, 16), K_train=4,
        width=256, depth=4, save_pred=None, cbatch=1024):
    torch.manual_seed(seed); np.random.seed(seed)
    consistency = 0.0
    if method.endswith("_c"):
        method_enc, consistency = method[:-2], 1.0
    else:
        method_enc = method
    B = load_image(img_name)
    tr = (min(train_scales),) if method == "pe_single" else tuple(train_scales)
    X, W, Y, A = build_train(B, tr)
    model = MLP(4 * L, width, depth)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda t: 0.1 ** (t / steps))
    mode = "pe" if method_enc in ("pe", "pe_single") else method_enc
    if method != "pe_ss":
        F = encode(X, W, L, mode)
    t0 = time.time(); n = len(X)
    for t in range(steps):
        idx = torch.randint(0, n, (batch,))
        if method == "pe_ss":
            xb, wb = X[idx], W[idx]
            jit = (torch.rand(batch, K_train, 2) - 0.5) * wb[:, None, :]
            pred = model(encode((xb[:, None, :] + jit).reshape(-1, 2), None, L, "pe")).reshape(batch, K_train, 3).mean(1)
        else:
            pred = model(F[idx])
        loss = (A[idx][:, None] * (pred - Y[idx]) ** 2).sum() / A[idx].sum() / 3
        if consistency > 0:
            # self-supervised refinement constraint: a box footprint is the exact union of its
            # 2x2 children, so  F(p, w) = mean_k F(p_k, w/2)  must hold for the true signal.
            wmin, wmax = 2 * min(train_scales) * BW, max(train_scales) * BW
            wp = torch.exp(torch.empty(cbatch, 1).uniform_(math.log(wmin), math.log(wmax)))
            pp = torch.empty(cbatch, 2).uniform_(-1, 1)
            off = torch.tensor([[-1, -1], [1, -1], [-1, 1], [1, 1]], dtype=torch.float32) * 0.25
            pc = (pp[:, None, :] + off[None] * wp[:, None, :]).reshape(-1, 2)
            wc = (wp / 2).repeat_interleave(4, 0)
            with torch.no_grad():
                child = model(encode(pc, wc, L, mode)).reshape(cbatch, 4, 3).mean(1)
            parent = model(encode(pp, wp, L, mode))
            loss = loss + consistency * ((parent - child) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    res = {"image": img_name, "method": method, "seed": seed, "L": L, "steps": steps,
           "train_scales": list(tr), "train_time_s": time.time() - t0, "eval": {}}
    preds = {}
    t1 = time.time()
    evals = [(s, 0, "ongrid") for s in train_scales] + [(s, s // 2, "shift") for s in test_scales]
    for s, o, kind in evals:
        gt, c, w = observe(B, s, o)
        te = time.time()
        p = predict(model, torch.from_numpy(c), torch.full((len(c), 1), w), L, method_enc).numpy().reshape(gt.shape)
        dt = time.time() - te
        key = f"{kind}_{s}"
        preds[key] = p; preds["gt_" + key] = gt
        res["eval"][key] = {"psnr": psnr(p, gt),
                            "ssim": float(structural_similarity(p, gt, channel_axis=2, data_range=1.0, win_size=7)),
                            "sec_per_mp": dt / (gt.shape[0] * gt.shape[1] / 1e6)}
    res["eval_time_s"] = time.time() - t1
    if save_pred:
        np.savez_compressed(save_pred, **preds)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--image"); ap.add_argument("--method"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=10); ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--train_scales", default="2,4,8,16")
    ap.add_argument("--tag", default="main"); ap.add_argument("--save_pred", default=None)
    a = ap.parse_args()
    r = run(a.image, a.method, a.seed, a.L, a.steps,
            train_scales=tuple(int(v) for v in a.train_scales.split(",")), save_pred=a.save_pred)
    r["tag"] = a.tag
    with open("results.jsonl", "a") as fh:
        fh.write(json.dumps(r) + "\n")
    print(json.dumps({k: (round(v["psnr"], 2)) for k, v in r["eval"].items()}), r["method"], round(r["train_time_s"]))
