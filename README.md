# vlm-mammo-decomposition

Code, study protocols and aggregate results for a decomposition of the gap between an end-to-end vision-language model (Qwen3-VL-8B-Instruct with LoRA) and a supervised head (the stage-1 head of a BI-RADS concept-bottleneck pipeline on frozen DINOv2 ViT-B/14) on mammographic calcifications and asymmetry. The repository accompanies a manuscript in preparation (title: "Training length, vision tower and read-out in the gap between a vision-language model and a supervised head in mammography").

Eight studies were run in sequence between 4 and 13 October 2026: s4v12, s4v3eval, s4diag, s4res, s6lc, s7zoom, s8 and s10 (two protocols, s10an and s10db, frozen together on 13 October after an external review of a draft reporting the first seven; `protocols/s10/`). s10db retrains the head for the number of optimiser updates of the VLM; s10an recomputes contrasts on the saved test scores (one examination per patient, probe against base-model read-outs, black images on the same breasts) and tabulates labels and composition; `code/s10/s10an_posthoc.py` and `s10db_posthoc.py` are the post hoc stratification by examination type (not part of the frozen protocols; see `protocols/s10/s10an_posthoc_ru.md`).

## Layout

| Folder | Content |
|---|---|
| `protocols/<study>/` | Written protocol of each study as it was frozen (in Russian), SHA-256 hash records taken on the compute server before the runs, and the deviation logs. |
| `code/<study>/` | Training, extraction, scoring and analysis scripts of each study, and the scripts that build the study reports. |
| `results/<study>/` | Aggregate output tables and logs of each study (AUC, contrasts, intervals, run metadata). |
| `data/paper4_numbers.csv` | The table of every number used in the manuscript, with its source. |
| `tools/` | Scripts that rebuild manuscript Tables 1, 2, S1-S7 and Figures 1-5, S1 and S2 from `data/` and `results/`. |

## What is not here

No images, patient identifiers, row-level scores, embeddings or trained weights are included. The EMBED licence requires written permission from Emory University for sharing weights, embeddings or row-level scores derived from EMBED. Only aggregate tables are provided.

## Data

The datasets used are available from their owners under their data-use terms: EMBED (Emory University), VinDr-Mammo (PhysioNet), CBIS-DDSM, CMMD and CDD-CESM (The Cancer Imaging Archive) and INbreast (from its authors).

## Protocol freezing

Each study had a written protocol with hypotheses, contrasts, decision rules, margins and thresholds. The protocol, the training code and the analysis code were frozen by recording SHA-256 hashes on the compute server before the corresponding runs. This is an internal freeze, not a public registration: the modification times of the hash records on the server (UTC+5) are the only time stamps. Protocols are kept in the language in which they were written; translating them would change the hashes. Deviations are listed in the `*_deviations.md` files. The manuscript states which studies were designed after earlier results were known.

## Reproducing

The scripts were run as they are on a shared GPU server, from `~/cbmammo` with `PYTHONPATH` set, and they use the server's cache layout and paths. Most of them import modules of the author's `cbmammo` package: the stage-1 pipeline (public at https://github.com/bcaitu/cbmammo-site-shift; the runs here used the version of that code present on the server on 4-13 October 2026) and the report-decoder modules (`cbmammo.reports`, `cbmammo.train_stage2*`), which are not part of this repository. The stage-1 modules that `lc_dtrain.py` and `lc_dtrain_budget.py` import (the head on cached DINOv2 features, the loss with masked unknown labels and the data manifests) are those of the stage-1 repository; the decoder modules are used only by the first two studies (report generation). Retraining therefore needs access to the controlled datasets and to those modules.

Software reported in the manuscript: Python 3.10.12, PyTorch 2.6.0 (CUDA 12.4), transformers 5.17.0, PEFT 0.21.0, scikit-learn 1.7.2, NumPy 2.2.6, SciPy 1.15.3; NVIDIA H200 GPUs.

The manuscript tables can be rebuilt without any dataset:

```
python3 tools/make_tables.py          # Tables 1, 2, S1, S2, S3  -> tables/
python3 tools/build_tables_extra.py   # Table S4                 -> tables/
python3 tools/build_table_s5.py       # Table S5 (seed-to-seed SD) -> tables/
python3 tools/build_tables_s6_s7.py   # Tables S6, S7 (recipes; updates, checkpoints, passes) -> tables/
python3 tools/make_fig1_design.py     # Figure 1                 -> figures/
python3 tools/make_fig2_fig3.py       # Figures 2 and 3          -> figures/
python3 tools/build_s10_numbers.py    # S10_* rows of data/paper4_numbers.csv from results/s10/
python3 tools/build_tables_s8_s11.py  # Tables S8-S11 (s10db, s10an, labels and composition, strata) -> tables/
python3 tools/make_fig4_gap.py        # Figure S3 (distance to D; Figure 4 of drafts 1-3) -> figures/
python3 tools/make_fig4_budget.py     # Figure 4 (head at the updates of the VLM) and Figure S4 (strata) -> figures/
python3 tools/fig5_resolution_place.py  # Figure 5               -> figures/
python3 tools/make_figS1.py           # Figures S1 and S2        -> figures/
```

`tools/extract_training_budget.py` reads the training logs on the compute server and wrote `results/s6lc/lc_training_budget.csv` (steps, selected checkpoints, epochs and rows per epoch of the V and D runs); it does not run without the server's `~/cbmammo` directory. `tools/build_tables_s6_s7.py` rewrites the `BUDGET_*` rows of `data/paper4_numbers.csv` from that file.

Python packages needed: pandas, numpy, matplotlib.

## Licence

Apache-2.0 (see `LICENSE`).
