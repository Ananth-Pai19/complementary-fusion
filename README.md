# complementary-fusion

This is the code implementation of the proposed complementary fusion architecture for post processing of OCT B scans.

The repository reproduces four classical OCT despeckling methods (TGV², FCFDS-NLM, PSO-wavelet and
K-SVD/OMP), compares them under one protocol, and then combines K-SVD and TGV in several ways:
cascaded, as a joint energy, and as a spatially varying fusion. All experiments use the Duke SD-OCT 2012
dataset that ships with the repository.

---

## Repository layout

| Folder | What it contains | Entry point |
|---|---|---|
| [`Sparsity_SDOCT_DATASET_2012/`](Sparsity_SDOCT_DATASET_2012/) | Duke SD-OCT 2012 dataset (Fang et al. 2012): 17 subjects with `*Raw Image.tif`, `*Averaged Image.tif` and the MSBTD benchmark `.mat` | [`Readme.txt`](Sparsity_SDOCT_DATASET_2012/Readme.txt) |
| [`TGV_denoising/`](TGV_denoising/) | Reproduction of Duan et al. (2016), second-order TGV decomposition (split Bregman), with baselines, tests and a report | [`README.md`](TGV_denoising/README.md), `scripts/00…07_*.py` |
| [`NLM_Denoising/`](NLM_Denoising/) | Non-local means with a fractional compact finite-difference kernel (Chen & Gao 2022) | `NLM_Denoising.ipynb` |
| [`PSO_Denoising/`](PSO_Denoising/) | Modified soft-threshold wavelet tuned by particle swarm optimisation (Subhedar et al. 2023) | `PSO_Denoising.ipynb` |
| [`KSVD_OMP/`](KSVD_OMP/) | K-SVD global dictionary + OMP with noise estimation (Zhang et al. 2022); holds the trained dictionary | `ksvd.ipynb` |
| [`Hybrid_PnP_Denoising/`](Hybrid_PnP_Denoising/) | PnP-ADMM hybrid: TGV² model step + NLM denoiser | `Hybrid_PnP_Denoising.ipynb` |
| [`Comparison/`](Comparison/) | The four methods behind one interface (`methods.py`), tuned under one protocol and scored with PSNR / SSIM / EPI | `comparison.ipynb` |
| [`Implementations/`](Implementations/) | Cascaded K-SVD → TGV² (adaptive variants A–F) and the early-stage joint K-SVD + TGV² energy | `cascaded_ksvd_tgv.ipynb`, `joint_ksvd_tgv_fusion.ipynb`, [`IMPLEMENTATIONS.md`](Implementations/IMPLEMENTATIONS.md) |
| [`KSVD_TGV_Fusion/`](KSVD_TGV_Fusion/) | Spatially varying pixel-wise fusion `X = w·X_K + (1−w)·X_T` with many weighting functions | `ksvd_tgv_fusion_analysis.ipynb` |

---

## Dependencies between folders

Most folders are self-contained and only read the dataset. Four of them reuse code or results from other
folders, so **keep the folder names as they are**:

```
Sparsity_SDOCT_DATASET_2012/  ◄── read by every notebook and by TGV_denoising/src/io_utils.py

KSVD_OMP/outputs/checkpoints/global_dict.npz (+ .json)   trained K-SVD dictionary
        ▲
        ├── Comparison/methods.py                     (KSVD_CHECKPOINT)
        ├── Implementations/cascaded_ksvd_tgv.ipynb   (DICT_PATH)
        └── Implementations/joint_ksvd_tgv_fusion.ipynb (DICT_PATH + global_dict.json)

TGV_denoising/src/  (tgv.py, operators.py, io_utils.py)  TGV² solver
        ▲
        ├── Comparison/methods.py                     (imports src.tgv.tgv_decompose)
        ├── Implementations/joint_ksvd_tgv_fusion.ipynb (loads TGV_denoising/src as a package)
        └── KSVD_TGV_Fusion/ksvd_tgv_fusion_analysis.ipynb (src.io_utils.noise_model_slope, via methods.TGV_DIR)

Comparison/methods.py  +  Comparison/outputs/
        ▲    (denoised/*.tif, metrics_per_image.csv, run_log.txt)
        └── KSVD_TGV_Fusion/ksvd_tgv_fusion_analysis.ipynb
```

`NLM_Denoising/`, `PSO_Denoising/` and `Hybrid_PnP_Denoising/` depend only on the dataset. The NLM and
wavelet code inside `Comparison/methods.py` is a port (a copy) of those notebooks, not an import.

**Where the dataset is looked up.** The notebooks resolve the dataset relative to their own folder
(`../Sparsity_SDOCT_DATASET_2012`). The `Implementations/` notebooks search upwards for it.
`NLM_Denoising`, `PSO_Denoising`, `Hybrid_PnP_Denoising` and `TGV_denoising` also accept an
`OCT_DUKE_DIR` environment variable pointing to a copy elsewhere.

---

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

`bm3d` (Hybrid PnP baseline) and `kagglehub` (PSO notebook on the Kermany dataset only) are optional.

## Running

Start Jupyter and **run each notebook from its own folder** (the default when you open it from the
Jupyter file browser, or with `jupyter nbconvert --execute`), because their paths are relative to that folder.
`KSVD_TGV_Fusion` can also be started from the repository root.

Suggested order when regenerating everything:

1. `KSVD_OMP/ksvd.ipynb`: trains and saves the global dictionary (already committed under `KSVD_OMP/outputs/checkpoints/`).
2. `TGV_denoising/`: see its [README](TGV_denoising/README.md) (`python -m pytest tests/ -q`, then `scripts/00…05`).
3. `NLM_Denoising`, `PSO_Denoising`, `Hybrid_PnP_Denoising`: independent, any order.
4. `Comparison/comparison.ipynb`: needs steps 1–2; writes `Comparison/outputs/`.
5. `Implementations/*.ipynb`: need steps 1–2.
6. `KSVD_TGV_Fusion/ksvd_tgv_fusion_analysis.ipynb`: needs step 4 (reuses its stored outputs; it regenerates any missing tuning cases).

Outputs that each notebook creates on first run (`outputs/`, `results_cascaded/`, `cache/`, `tables/`, …)
are written inside that notebook's folder.

---

## Methods and references

| Method | Paper |
|---|---|
| TGV² decomposition | Duan et al., *Biomedical Signal Processing and Control* 24 (2016) 120–127 |
| FCFDS non-local means | Chen & Gao, *Micromachines* 13(12) (2022) 2039 |
| PSO modified soft-threshold wavelet | Subhedar et al., *Traitement du Signal* 40(3) (2023) 1179–1185 |
| K-SVD + OMP, global dictionary | Zhang et al., *Optics Express* 30(4) (2022) 5788–5802 |
| MSBTD benchmark / dataset | Fang et al., *Biomedical Optics Express* 3(5) (2012) 927–942 |

See [`Comparison/readme.md`](Comparison/readme.md) for the full citations and
[`Implementations/IMPLEMENTATIONS.md`](Implementations/IMPLEMENTATIONS.md) for the results, protocols and trade-offs.
