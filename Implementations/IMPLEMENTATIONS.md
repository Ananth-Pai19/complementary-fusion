# Implementations, results and trade-offs — OCT B-scan denoising

This document inventories every denoiser implemented in this repository, states the protocol each
number comes from, and compares them. It is written to be read alongside:

* `Implementations/joint_ksvd_tgv_fusion.ipynb` — the early-stage joint energy method (new),
* `Implementations/results_joint_energy/` — its tables and figures,
* `Implementations/cascaded_ksvd_tgv.ipynb` — the adaptive (cascaded) K-SVD+TGV study,
* `Comparison/outputs/` — the tuned four-method comparison.

The earlier write-ups `RESULTS_SUMMARY.md` (adaptive K-SVD+TGV) and `JOINT_FEASIBILITY_REPORT.md`
(alternating-loop feasibility), and their `adaptive_oct/` code, are **not part of this repository**.
Their numbers are kept below for reference only. The adaptive pipeline itself is now
`cascaded_ksvd_tgv.ipynb`.

---

## 0. Repository map

All paths are relative to the repository root.

| folder / file | method | entry point | results |
|---|---|---|---|
| `KSVD_OMP/` | K-SVD + OMP sparse coding (Zhang et al. 2022) | `ksvd.ipynb`, `outputs/checkpoints/global_dict.npz` | `KSVD_OMP/outputs/table3_real_data.csv` |
| `TGV_denoising/` | second-order TGV decomposition (Duan et al. 2016) | `scripts/00…07_*.py`, `src/tgv.py` | `TGV_denoising/results/` |
| `NLM_Denoising/` | FCFDS-kernel non-local means (Chen & Gao 2022) | `NLM_Denoising.ipynb` | `NLM_Denoising/outputs/duke_nlm/` (created on run) |
| `PSO_Denoising/` | modified-soft-threshold wavelet tuned by PSO (Subhedar et al. 2023) | `PSO_Denoising.ipynb` | `PSO_Denoising/outputs/duke/` (created on run) |
| `Hybrid_PnP_Denoising/` | PnP-ADMM hybrid: TGV² model step + non-local-means denoiser (NLM / TGV / BM3D baselines) | `Hybrid_PnP_Denoising.ipynb` | `Hybrid_PnP_Denoising/outputs/` (created on run) |
| `Comparison/` | the four methods behind one interface + tuned benchmark | `comparison.ipynb`, `methods.py` | `Comparison/outputs/metrics_per_image.csv`, `summary_*.csv` |
| `Implementations/cascaded_ksvd_tgv.ipynb` | K-SVD then TGV with spatially varying weights (methods A–F) | the notebook | `Implementations/results_cascaded/` (created on run) |
| `Implementations/joint_ksvd_tgv_fusion.ipynb` | **early-stage joint energy** (K-SVD prior + TGV² in one objective) | the notebook | `Implementations/results_joint_energy/` |
| `KSVD_TGV_Fusion/` | spatially varying pixel-wise fusion of K-SVD and TGV outputs | `ksvd_tgv_fusion_analysis.ipynb` | `KSVD_TGV_Fusion/figures/` |
| `Sparsity_SDOCT_DATASET_2012/` | 17 paired raw / averaged SDOCT B-scans | — | — |

---

## 1. Evaluation protocol (read this before comparing numbers)

**Data.** 17 subjects, each with a single-shot `*Raw Image.tif` (the noisy input **Y**) and a
registered multi-frame `*Averaged Image.tif` (the reference **R**). All images are 8-bit,
≈ 530 × 970 px.

**Reference caveat.** The averaged image is not ground truth: it has its own noise floor
(≈ 4 grey levels) and residual motion blur. PSNR therefore saturates well below infinity and the
reference's own edge width is an upper bound on the true optical edge (5.9 px on subject 13).

**Metrics.** PSNR (`10 log10(255²/MSE)`), SSIM (Wang 2004, 11×11 Gaussian, σ=1.5, data_range 255),
EPI (correlation of Sobel gradient magnitudes). The joint notebook adds ROI PSNR (tissue mask from
Otsu on the reference), edge-band SSIM (SSIM averaged inside the dilated reference edge band) and
the 10–90 % boundary rise distance in px (larger = softer edge).

**Two benchmarks exist in the repository and they must not be mixed:**

1. **17-subject protocol, repository-frozen settings** — used by `results_adaptive/`,
   `results_joint/` (neither is included in this repository) and `results_joint_energy/`. All rows in §3.1 share this protocol, so they are
   directly comparable to each other.
2. **13-image protocol with per-method tuning** — `Comparison/outputs/`, where each method's
   strength parameters were grid-tuned on 4 images before scoring 13 evaluation images. This yields
   higher absolute numbers (§3.2) and is the right table for "each method at its best", but it is a
   different image subset and a different tuning budget.

---

## 2. The methods

### 2.1 K-SVD + OMP sparse coding (Zhang et al. 2022)

**Idea.** Every 8×8 patch of the image is approximated by a few atoms of a dictionary learned once
from high-quality (averaged) OCT patches. Denoising = reconstruct each patch from its sparse code
and average the overlapping reconstructions.

**Objective / algorithm.**
`min_{X,α} λ‖X−Y‖² + Σ_ij ‖R_ij X − D α_ij‖² + Σ_ij μ_ij‖α_ij‖₀`, solved by alternating OMP
(sparse coding, error limit `ε = C σ √64`, at most 12 atoms) and the closed-form image update
`X = (λI + ΣRᵀR)⁻¹(λY + ΣRᵀDα)` with `λ = 30/σ`. The noise level σ is estimated per image from
the weakest-texture 64×64 tile (GLCM entropy) using the Immerkær Laplacian estimator.

**Settings.** `D`: 64×256, trained on 30 000 patches from averaged images of subjects 1–12, 20 K-SVD
sweeps, sparsity 6. Paper defaults `C=1.15`, `λ_scale=30`; tuned `C=1.6`, `λ_scale=3`.

**Results.** 17 subjects, defaults: **26.11 ± 1.77 dB**, SSIM 0.551, EPI 0.452, ≈7 s/frame.
Tuned (13-image subset): **28.01 dB**, SSIM 0.675, EPI 0.635 — a **+1.11 dB** gain over the defaults,
mainly from the weaker fidelity weight (`λ_scale 30 → 3`).

**Advantages.** Interpretable generative prior; the dictionary is learned once and reused, so it is
fast per image; no iteration to convergence; good at restoring layer texture and fine structure.
The dictionary file is small (124 KB on disk: 256 atoms × 64 coefficients).

**Disadvantages.** Very sensitive to its two strength parameters (1.1 dB between "paper" and
"tuned"); OMP over 514 k patches is the runtime bottleneck; leaves visible residual speckle at the
default setting; block artefacts possible at strong compression; the L0/OMP step is non-convex and
the codes are unstable across iterations (see 2.8).

### 2.2 Second-order TGV decomposition (Duan et al. 2016)

**Idea.** Decompose the log-domain image into a piecewise-smooth structure `u` and an oscillatory
texture `g`, penalising first and second derivatives of `u`. Second order avoids the staircase
artefact of plain TV on smooth intensity ramps.

**Objective.** `min_{u,g,p} ½‖f−u−div g‖² + β‖g‖₁ + α₁‖∇u−p‖₁ + α₀‖ℰ(p)‖₁`, solved by
split Bregman with three FFT closed-form updates (300 iterations max, tol 1e-3; typically 15).

**Settings.** `α₀=α₁=0.1`, `β=0.1`, `λ₁=λ₂=λ₃=2`; tuned `α=0.2`, `β=1.0`.

**Results.** 17 subjects, defaults: **27.26 ± 1.75 dB**, SSIM 0.670, EPI 0.574, ≈6 s/frame (15
iterations). Tuned (13-image subset): **27.92 dB**, SSIM 0.679, EPI 0.617.

**Advantages.** Best structural similarity and edge preservation among the single-pass methods;
explicitly models the noise as a separate texture field, so it can remove speckle without
sacrificing layer contrast; few iterations; no dictionary; robust across subjects (lowest s.d.).
Its edge width (6.7 px) is close to the reference's own (5.9 px).

**Disadvantages.** Assumes additive noise, so it must be applied in the log domain; tends to erase
the finest intra-layer texture (the "plastic" look); the texture weight β is another
sensitive parameter; cannot recover detail it has already smoothed away.

### 2.3 Non-local means with a fractional compact finite-difference kernel (Chen & Gao 2022)

**Idea.** Average pixels whose neighbourhoods look similar, with the similarity kernel shaped by a
fractional-order edge detector so that smoothing stops at edges.

**Settings.** 21×21 search, 7×7 patch, fractional order α=0.5 (tuned 0.3), 4 taps, `h = h_factor ·
σ_MAD · √(kernel mass)` with `h_factor` tuned 2.4.

**Results.** Tuned 13-image subset: **27.95 dB**, SSIM 0.662, EPI 0.618. Paper defaults:
27.71 dB.

**Advantages.** No dictionary, no tuning of a global strength beyond `h`; preserves texture
statistics (it averages real samples, it does not synthesise); performs well on the tuning subset
without any learned component; conceptually simple.

**Disadvantages.** Expensive per pixel (441 offset comparisons, ≈3 s/frame in the tuned comparison
— comparable to TGV there, but it scales with the search window); the 21×21 window is a hard
locality limit; the noise-level estimate must be right or `h` is wrong; SSIM slightly below
TGV/K-SVD because it leaves more residual grain.

### 2.4 Modified soft-threshold wavelet + PSO (Subhedar et al. 2023)

**Idea.** Threshold the 2-D wavelet detail bands with a modified soft rule that keeps a small
fraction of small coefficients (unlike plain soft thresholding), and let PSO choose the threshold
parameters.

**Settings.** `db2`, level 6 (tuned), parameters `(α, β, K)` from PSO (30 particles, 15 iterations,
108 s); universal threshold scaled by the MAD noise estimate.

**Results.** Tuned 13-image subset: **22.43 dB**, SSIM 0.596, EPI 0.295. Original settings:
19.17 dB — i.e. the method is extremely parameter-sensitive (+3.3 dB from tuning), and even tuned it
is **5.5 dB below** the other methods, with by far the worst edge preservation.

**Advantages.** Fast once tuned; no dictionary, no iterative solve; the modified threshold avoids
the over-smoothing of plain soft thresholding; useful as a cheap baseline.

**Disadvantages.** Clearly the weakest method here on this data (leakage/ringing on sharp retinal
boundaries); the PSO tuning costs more than the denoising itself and does not transfer well across
images; level 6 with `db2` is aggressive.

### 2.5 MSBTD (Fang et al. 2012) — external reference

The dataset ships the authors' own multi-scale sparsity result (`*_MSBTD result.mat`). It is an
external method, not implemented here, and is included for calibration only: **27.38 dB**,
SSIM 0.589, EPI 0.596 on the 13-image subset. Note that the 2012 method is competitive with the
2016–2023 implementations on PSNR, which says more about the ceiling imposed by the reference than
about the newer methods.

### 2.6 Late-stage fusion (pixel-space blending)

**Idea.** Run K-SVD and TGV independently and average their outputs:
`X_blend = w·X_KSVD + (1−w)·X_TGV`.

**Settings.** `w` selected on subjects 13–14 by mean PSNR. On these scans the optimum is `w* = 0`
(K-SVD is ≈1 dB below TGV), so the tuned blend collapses to TGV; the equal-weight `w=0.5` blend is
reported as the non-degenerate baseline.

**Results.** `w=0.5`: **26.96 dB**, SSIM 0.625, EPI 0.532. Tuned (`w*=0`): 27.26 dB — identical to
TGV, i.e. **late fusion adds nothing**.

**Advantages.** Trivial to implement and to parallelise; can be applied to any pair of denoisers;
useful as a sanity-check baseline.

**Disadvantages.** Mixing a noisy image with a smooth one can only trade noise for blur; there is no
mechanism to keep the K-SVD texture where it is right and the TGV smoothness where it is right;
the optimal weight is image-dependent, so a single `w` is always wrong for some scans.

### 2.7 Cascade K-SVD → TGV, and spatially adaptive variants (methods A–F)

**Idea.** Denoise with K-SVD, then feed the result to TGV², optionally letting the TGV weight vary
per pixel: `α(x) = α_min + (α_max−α_min)(1−M(x))`, with `M` a structure map (gradient,
structure tensor, depth, K-SVD byproducts, or an oracle derived from the reference).

**Results (17 subjects).**

| ID | method | PSNR | SSIM | EPI |
|---|---|---|---|---|
| A | K-SVD alone | 26.11 ± 1.82 | 0.55 | 0.45 |
| B | K-SVD + constant-α TGV | **27.52 ± 1.86** | 0.682 | 0.581 |
| C | + gradient-guided TGV | 27.52 ± 1.85 | 0.68 | 0.59 |
| D | + structure-tensor-guided | 27.54 ± 1.88 | 0.68 | 0.60 |
| E | + K-SVD-metadata-guided | 27.53 ± 1.85 | 0.68 | 0.59 |
| F | + oracle-guided (upper bound) | **27.59 ± 1.86** | 0.68 | 0.63 |

**Advantages.** The strongest PSNR in the repository (27.5–27.6 dB) with modest cost (≈5 s for the
TGV stage on a cached K-SVD output); the adaptive variants buy edge fidelity (EPI 0.58 → 0.63) at no
PSNR cost; the oracle row quantifies how much a perfect structure map could add (+0.07 dB PSNR,
+0.05 EPI).

**Disadvantages.** Two stages, two tuning problems, error propagation from stage 1 to stage 2; the
adaptive versions need a structure map that correlates with the true edges — the best non-oracle map
(gradient) reaches only r ≈ 0.70 on the edge band, so most of the oracle gain is not realisable;
still smoother than K-SVD (rise 6.7 px).

### 2.8 Alternating K-SVD/TGV feedback loop (feasibility study)

**Idea.** Iterate the two stages five times, feeding the TGV output back into the patch coder:
`P ← aggregate(D·OMP(X))`, `f ← (λY + γP)/(λ+γ)`, `X ← TGV(f)`, all in the log domain.

**Results.** **27.52 dB**, SSIM 0.680, EPI 0.544, 53 s/frame — i.e. it reproduces the one-pass
cascade exactly, at 11× the runtime. Gate analysis: energy decreases, residual is white, but the
OMP codes cycle between iterations (mean Jaccard 0.21) and the gain over its own first iteration is
only +0.24 dB.

**Advantages.** Simple to bolt onto an existing pipeline; no new solver.

**Disadvantages.** The feedback does not add information: each pass re-codes an already-smoothed
image, so the fixed point is the cascade's output. The dictionary is applied in a domain it was not
trained in, and the L0 codes are unstable. Not worth the 11× cost.

### 2.9 Early-stage joint energy — this work (`joint_ksvd_tgv_fusion.ipynb`)

**Idea.** Optimise **one** image `X` under the data term, the TGV² penalty and the K-SVD patch
prior simultaneously, instead of blending two outputs:

```
min_{X,α,p} ½‖Y−X‖² + α₁‖∇X−p‖₁ + α₀‖ℰ(p)‖₁
            + λ Σ_ij ( ‖R_ij X − D α_ij‖² + μ‖α_ij‖₀ )
```

**Algorithm (alternating minimisation + split Bregman).**

1. **α-block** — batched OMP of every patch of the current `X` on the fixed dictionary `D`
   (L0 budget T = 1 atom), reconstruct `X̂_p = avg_ij(Dα_ij)`.
2. **X-block** — exact FFT solve (patches wrap around, stride 2 ⇒ ΣRᵀR = 16 I):
   `(1 + γ₁|∇|² + 2λC) X̂ = FFT(Y − γ₁ div(d+p−b_d) + 2λC X̂_p)`.
3. **d-block** — `d = shrink(∇X − p + b_d, α₁/γ₁)`.
4. **p-block** — FFT solve of `(γ₁I + γ₂ℰᵀℰ)p = γ₁(∇X − d + b_d) − γ₂div_ℰ(v − b_v)`.
5. **v-block** — `v = shrink(ℰ(p) + b_v, α₀/γ₂)` with weights (1,1,2).
6. **Bregman updates** and energy logging; stop when `‖X_k − X_{k−1}‖/‖X_k‖ < 10⁻⁵` (or 40 iters).

**Settings.** Working domain `log(1+Y/255)/log 2`; `λ=0.10`, `α₁=0.08`, `α₀=0.04` (selected on
subjects 13–14 from a 6-point grid), `γ₁=γ₂=2`, `T=1`, patch 8, stride 2. Converges in 39–40
iterations, ≈ 42 s/frame; the OMP block dominates (roughly two-thirds of each iteration).

**Results (17 subjects).** **27.27 ± 1.78 dB**, ROI PSNR 24.70, SSIM 0.639, edge-band SSIM 0.624,
EPI 0.421, rise 6.98 px.

**Advantages.** A single objective with one tuning problem instead of two stages + a blend weight;
explicitly balances data, TGV smoothness and dictionary representation; beats K-SVD by **+1.16 dB**
(13/17 subjects) and equal-weight late fusion by **+0.32 dB** (12/17); gives the lowest MAE/RMS on
the featured subject (9.37/13.04 vs TGV 9.57/13.67); the FFT X-block makes each iteration cheap; the
energy and its three terms are logged, so the trade-off is inspectable.

**Disadvantages.** PSNR is a **tie with plain TGV²** (27.27 vs 27.26; +0.01 dB mean, 7/17 wins,
median −0.19 dB) and **below the tuned cascade** (27.52); SSIM, edge SSIM and EPI are all below TGV;
edge softening is **not** eliminated (rise 6.98 vs TGV 6.71 vs reference 5.91); 42 s/frame is ≈7×
TGV and ≈6× one K-SVD pass; the dictionary was trained on linear 8-bit patches but is applied in the
log domain with a one-atom budget, which limits what the patch prior can contribute; the data term
has no texture variable (unlike Duan's `½‖f−u−div g‖²+β‖g‖₁`), so noise must be absorbed by `X` or
smoothed away.

---

## 3. Results

### 3.1 17-subject protocol (repository-frozen settings)

| method | PSNR (dB) | SSIM | EPI | rise (px) | time/frame |
|---|---|---|---|---|---|
| raw input | 17.74 ± 0.37 | 0.079 | 0.125 | 3.80* | — |
| K-SVD (paper defaults) | 26.11 ± 1.77 | 0.551 | 0.452 | 5.82 | 7 s |
| TGV² (α=0.1, β=0.1) | 27.26 ± 1.75 | 0.670 | 0.574 | 6.71 | 6 s |
| late blend, tuned (w*=0) | 27.26 ± 1.75 | 0.670 | 0.574 | 6.71 | 13 s |
| late blend, equal (w=0.5) | 26.96 ± 1.74 | 0.625 | 0.532 | 6.21 | 13 s |
| cascade K-SVD→TGV (B) | **27.52 ± 1.86** | **0.682** | 0.581 | — | ≈12 s (7 + 5) |
| cascade, structure-guided (D) | 27.54 ± 1.88 | 0.68 | 0.60 | — | ≈14 s |
| cascade, oracle (F) | **27.59 ± 1.86** | 0.68 | 0.63 | — | ≈14 s |
| alternating feedback (5 iters) | 27.52 ± 1.92 | 0.680 | 0.544 | — | 53 s |
| **joint energy (this work)** | 27.27 ± 1.78 | 0.639 | 0.421 | 6.98 | 42 s |

`*` the raw scan's rise distance is noise-inflated (noise creates steep local crossings) and is not
a sharpness win; the reference's own rise distance is 5.91 px on subject 13.

**Paired PSNR of the joint estimate:** +1.16 dB vs K-SVD (13/17 wins), +0.32 dB vs equal blend
(12/17), +0.01 dB vs TGV² (7/17), +0.01 vs tuned blend (7/17).

### 3.2 Tuned 13-image comparison (`Comparison/outputs/`)

| method | original settings | tuned PSNR | SSIM | EPI |
|---|---|---|---|---|
| noisy | — | 17.79 | 0.083 | 0.153 |
| K-SVD | 26.90 | **28.01** | 0.675 | 0.635 |
| NLM (FCFDS) | 27.71 | 27.95 | 0.662 | 0.618 |
| TGV² | 27.81 | 27.92 | 0.679 | 0.617 |
| MSBTD (2012, external) | — | 27.38 | 0.589 | 0.596 |
| Wavelet + PSO | 19.17 | 22.43 | 0.596 | 0.295 |

Tuned settings: K-SVD `C=1.6, λ_scale=3`; TGV `α=0.2, β=1.0`; NLM `h_factor=2.4, α=0.3`;
wavelet level 6 with PSO `(α,β,K)`.

**Take-away:** with per-method tuning, K-SVD, NLM and TGV² land within 0.1 dB of each other
(28.01 / 27.95 / 27.92). On this data the choice between them is about *which error* you care
about (texture vs smoothness vs runtime), not about PSNR. The wavelet baseline is 5.5 dB behind and
the 2012 MSBTD result is still competitive — the reference image, not the algorithm, is the binding
constraint.

---

## 4. Advantages / disadvantages at a glance

| method | strengths | weaknesses | best for |
|---|---|---|---|
| K-SVD | restores texture; small fixed dictionary; interpretable | parameter-sensitive (1.1 dB); slow OMP; residual speckle | texture-critical work; dictionary reuse across scanners |
| TGV² | best SSIM/EPI per pass; few iterations; robust s.d. | "plastic" look; erases fine texture; log domain required | clean, fast structural denoising; baseline for fusion |
| NLM | no training; preserves real sample statistics | 441 offsets/pixel; window-limited; leaves grain | when no dictionary exists; different noise physics |
| Wavelet+PSO | cheap; simple | weakest here (−5.5 dB); tuning does not transfer | quick previews, not quantitative work |
| late blend | trivial; parallelisable | cannot beat its best component; image-dependent weight | sanity check only |
| cascade / adaptive | best PSNR (27.5–27.6); EPI up to 0.63 | two-stage tuning; error propagation; needs structure maps | when PSNR/edge fidelity matters most |
| alternating feedback | simple to add | no gain over one pass; 11× cost; unstable codes | not recommended |
| **joint energy** | one objective; beats K-SVD by 1.16 dB; inspectable energy; exact FFT X-step | ties TGV, below cascade; lower SSIM/EPI; 7× cost; no edge-softening gain | research baseline for principled multi-prior fusion |

---

## 5. Practical guidance

* **Need the best PSNR now** → K-SVD → TGV cascade (method B), or the structure-guided variant if
  edge fidelity matters (D/F). 27.5–27.6 dB at ≈12–14 s/frame end to end.
* **Need the best speed/quality ratio** → TGV² alone (27.26 dB, 6 s, no dictionary).
* **Need texture, not smoothness** → K-SVD (tuned: 28.0 dB on the 13-image subset) or NLM.
* **Need one model with an explicit, inspectable trade-off** → the joint energy; expect
  TGV-level PSNR, better than K-SVD (at its repository settings) and late fusion, at 42 s/frame.
* **Do not** use late pixel blending (it collapses to its best component) or the alternating
  feedback loop (same output as the cascade at 11× cost) unless the point is the comparison itself.
* **Do not** compare numbers across §3.1 and §3.2 without noting the different subsets and tuning
  budgets.

---

## 6. Why the joint energy lands where it does

1. **Two priors, one landscape.** TGV wants piecewise-smooth structure; the one-atom dictionary
   prior wants every patch to be a single learned pattern. Their intersection is smoother than
   K-SVD and no sharper than TGV — hence the PSNR tie with TGV and the rise distance of 6.98 px.
2. **The data term has no texture variable.** Duan's model gives noise its own field `g`; here
   noise must be explained by `X` or removed by the TGV seminorm, which pushes the solution toward
   smoothing whenever α₁ is large enough to denoise.
3. **The dictionary is used out of domain.** It was trained on linear 8-bit patches; the solver
   works in the normalised log domain with a one-atom budget. OMP is scale-equivariant, so the
   support transfer is valid, but a dictionary trained in the working domain (or a larger budget
   with a noise-matched μ) would give the patch block more to say.
4. **The tuning grid was narrow.** α₁ was explored up to 0.08 while the tuned standalone TGV uses
   α ≈ 0.2 and the cascade reaches 27.52 dB; the joint's PSNR ceiling was therefore not fully
   explored.
5. **Cost is structural.** One OMP pass over 128 k patches per outer iteration dominates; the FFT
   X-block is negligible. Reaching the cascade's quality would need either fewer, better-targeted
   patch passes or a GPU OMP.

---

## 7. Reproduction

```bash
# from the repository root, after: pip install -r requirements.txt
# (each notebook is executed from its own folder, so its relative paths resolve)

# joint energy (this work) — executes the notebook end to end (~20 min, CPU)
jupyter nbconvert --to notebook --execute --inplace Implementations/joint_ksvd_tgv_fusion.ipynb

# adaptive (cascaded) K-SVD + TGV (methods A–F, figures)
jupyter nbconvert --to notebook --execute --inplace Implementations/cascaded_ksvd_tgv.ipynb

# tuned four-method comparison
jupyter nbconvert --to notebook --execute --inplace Comparison/comparison.ipynb

# K-SVD dictionary training / denoising notebook
jupyter nbconvert --to notebook --execute --inplace KSVD_OMP/ksvd.ipynb

# TGV reproduction pipeline (scripts, see TGV_denoising/README.md)
cd TGV_denoising && python -m pytest tests/ -q && python scripts/04_real_data.py
```

The alternating-feedback feasibility study (`run_joint_feasibility.py`) is not included in this
repository.

The joint notebook is deterministic (no RNG); re-running reproduces the tables and figures in
`results_joint_energy/`.

---

## 8. Caveats that apply to every row

* The averaged reference is a registered multi-frame average, not ground truth: PSNR saturates and
  the reference's edge width (5.9 px) is an upper bound, not the optical truth.
* Raw-vs-reference pairs are different acquisitions; sub-pixel motion means high-frequency content
  cannot match pixel-for-pixel, so all methods share an irreducible error floor.
* Hyper-parameters differ per study (each was tuned on its own validation split). The 17-subject
  table uses repository-frozen settings; the 13-image table uses per-method tuning. Never mix them.
* Timings are wall-clock on an 8-core CPU with NumPy/SciPy only; a GPU would change the ranking of
  the patch-based methods (K-SVD, joint) relative to the FFT-based ones (TGV).
