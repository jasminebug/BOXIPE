# Box-IPE / Lobe-IPE + Refinement Consistency — reproduction code

Code for the paper "Exact Box-Footprint Encodings and Refinement Consistency for Anti-Aliased Multiscale Neural Fields".

## Requirements
Python 3.10+, `torch`, `numpy`, `scipy`, `scikit-image`, `matplotlib`. CPU is enough (all paper results were produced on a 2-core CPU).

## Files
- `fipe.py` — encodings (PE, Gaussian IPE, Box-IPE, Lobe-IPE), supersampling baseline, refinement-consistency loss (`*_c` methods), training and the shifted-grid evaluation protocol. One run = one image × one method × one seed.
- `runner.py` — runs the full experiment grid (162 runs, resumable; skips runs already in `results.jsonl`).
- `analyze.py` — aggregates `results.jsonl` into `stats.json` (tables, Wilcoxon tests) and draws Figs. 2, 4, 5.
- `theory.py` — Fig. 1 (gain curves) and the Monte Carlo check of Proposition 2.
- `qual.py` — Fig. 3 (qualitative crops), e.g. `python3 qual.py rocket 0.5 0.45`.
- `eqs.py` — renders the equations used in the manuscript.
- `results.jsonl`, `stats.json` — raw per-run results and aggregated statistics used in the paper.

## Single run example
```
python3 fipe.py --image astronaut --method ipe_hyb_c --seed 0 --tag main
```
Methods: `pe_single`, `pe`, `pe_ss`, `ipe_gauss`, `ipe_box`, `ipe_hyb` (Lobe-IPE), `ipe_gauss_c`, `ipe_hyb_c` (Lobe-IPE + C).
Use `--train_scales 2,16` for the sparse protocol and `--L` for the frequency sweep.
