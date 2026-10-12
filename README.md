# vlm-mammo-decomposition

Code, study protocols and aggregate results for a controlled decomposition of why an end-to-end vision-language model (Qwen3-VL-8B-Instruct with LoRA) trails a supervised specialist head (the stage-1 head of a BI-RADS concept-bottleneck pipeline on frozen DINOv2 ViT-B/14) on subtle mammographic findings. The repository accompanies a manuscript in preparation (title: "Why an end-to-end vision-language model trails a specialist head on subtle mammographic findings: a controlled decomposition").

Seven studies were run in sequence between 4 and 9 October 2026: s4v12, s4v3eval, s4diag, s4res, s6lc, s7zoom and s8.

## Layout

| Folder | Content |
|---|---|
| `protocols/<study>/` | Written protocol of each study as it was frozen (in Russian), SHA-256 hash records taken on the compute server before the runs, and the deviation logs. |
| `code/<study>/` | Training, extraction, scoring and analysis scripts of each study, and the scripts that build the study reports. |
| `results/<study>/` | Aggregate output tables and logs of each study (AUC, contrasts, intervals, run metadata). |
| `data/paper4_numbers.csv` | The table of every number used in the manuscript, with its source. |
| `tools/` | Scripts that rebuild manuscript Tables 1, 2, S1-S4 and Figures 5 and S1 from `data/` and `results/`. |

## What is not here

No images, patient identifiers, row-level scores, embeddings or trained weights are included. The EMBED licence requires written permission from Emory University for sharing weights, embeddings or row-level scores derived from EMBED. Only aggregate tables are provided.

## Data

The datasets used are available from their owners under their data-use terms: EMBED (Emory University), VinDr-Mammo (PhysioNet), CBIS-DDSM, CMMD and CDD-CESM (The Cancer Imaging Archive) and INbreast (from its authors).

## Protocol freezing

Each study had a written protocol with hypotheses, contrasts, decision rules, margins and thresholds. The protocol, the training code and the analysis code were frozen by recording SHA-256 hashes on the compute server before the corresponding runs. This is an internal freeze, not a public registration: the modification times of the hash records on the server (UTC+5) are the only time stamps. Protocols are kept in the language in which they were written; translating them would change the hashes. Deviations are listed in the `*_deviations.md` files. The manuscript states which studies were designed after earlier results were known.

## Reproducing

The scripts were run as they are on a shared GPU server, from `~/cbmammo` with `PYTHONPATH` set, and they use the server's cache layout and paths. Most of them import modules of the author's `cbmammo` package: the stage-1 pipeline (also in https://github.com/bcaitu/cbmammo-site-shift) and the report-decoder modules (`cbmammo.reports`, `cbmammo.train_stage2*`), which are not part of this repository. Retraining therefore needs access to the controlled datasets and to those modules.

Software reported in the manuscript: Python 3.10.12, PyTorch 2.6.0 (CUDA 12.4), transformers 5.17.0, PEFT 0.21.0, scikit-learn 1.7.2, NumPy 2.2.6, SciPy 1.15.3; NVIDIA H200 GPUs.

The manuscript tables can be rebuilt without any dataset:

```
python3 tools/make_tables.py          # Tables 1, 2, S1, S2, S3  -> tables/
python3 tools/build_tables_extra.py   # Table S4                 -> tables/
python3 tools/fig5_resolution_place.py  # Figure 5               -> figures/
python3 tools/make_figS1.py           # Figure S1                -> figures/
```

Python packages needed: pandas, numpy, matplotlib.

## Licence

Apache-2.0 (see `LICENSE`).
