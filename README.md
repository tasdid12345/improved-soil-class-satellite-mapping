# Class-balanced LightGBM with neighbourhood features vs random forest for land/soil classes

Motivation: Soil classes are increasingly mapped from multispectral satellite imagery, and the paper "Soil classification based on Sentinel-2 Products using HorusApp application" (2019) describes an operational pipeline for this task. Its exact model and workflow are not described in enough detail to reimplement, so this experiment asks a narrower, reproducible question on a public multispectral dataset.

Question: Can a class-balanced LightGBM model with spectral-neighbourhood features beat a standard random forest at classifying land and soil surface classes from multispectral Sentinel-2-like pixels?

Dataset: OpenML 182 (Statlog Landsat Satellite), six surface classes including several soil types, with 3x3 pixel neighbourhoods of four spectral bands. It is a public stand-in for Sentinel-2 soil-class mapping and is not the paper's own dataset. A stratified 75/25 split with seed 42 is used.

Baseline: a standard random forest on the raw 36 spectral features.

Improved model: a class-balanced LightGBM on the raw features plus per-band 3x3 neighbourhood means, standard deviations and band ratios. Its hyper-parameters are tuned with 3-fold cross-validation on the training split only.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: results are written to `results.json` (and a chart to `figures/`).

## Results (latest run)

On the OpenML Statlog Landsat Satellite dataset (single held-out test set, n=1,608), a standard random-forest pixel classifier reached 0.901 macro-F1, while my class-balanced LightGBM with 3x3 neighbourhood features reached 0.923, a modest gain of about 2.2 points.

| Metric | Value |
|---|---|
| baseline_score | 0.9014 |
| improved_score | 0.9231 |
| improvement | 0.02167 |
| baseline_score_std | 0.001163 |
| improved_score_std | 0.0009305 |
| baseline_accuracy | 0.9196 |
| improved_accuracy | 0.9359 |
| baseline_balanced_accuracy | 0.8943 |
| improved_balanced_accuracy | 0.919 |
| lgbm_cv_macro_f1 | 0.9119 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Stand-in for Sentinel-2 soil mapping, not the paper's HorusApp pipeline or data. Assumes OpenML features are pixel-major (9 pixels x 4 bands) for the neighbourhood features. Only the improved model is tuned; the baseline uses default RF settings, so the comparison is not budget-matched in tuning. Single fixed split, so test estimates have sampling uncertainty; std reflects model seed only. The openml package's own retry/timeout settings are used for download.

## Reference

This project is motivated by the following paper:

> V Bacu, T Stefanut, C Nandra (2019). *Soil classification based on Sentinel-2 Products using HorusApp application.*. Geophysical. https://search.ebscohost.com/login.aspx?direct=true&profile=ehost&scope=site&authtype=crawler&jrnl=10297006&AN=140482556&h=eGJH84AYGxDfc%2FRqNvcLMETi%2FjCPB8Daoly%2BbJstidPOpOp%2BWgS64rhmC5JOnEzfLJD%2F3ukAW8%2BjR5NckHsmEw%3D%3D&crl=c

Paper author: Prof. Dorian Gorgan, Technical University of Cluj-Napoca (dorian.gorgan@cs.utcluj.ro)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
