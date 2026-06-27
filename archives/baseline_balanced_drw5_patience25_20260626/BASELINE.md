# Baseline: balanced_drw5_patience25

## Scope

This archive freezes the current 5-fold Stage-2 baseline.

- Date: 2026-06-26
- Config: `configs/stage2_finetune.yaml`
- Output source: `checkpoints/stage2_cv`
- Split: 5 held-out validation folds from `data_processed/messidor/folds.csv`
- Head: CORN
- Attention: cross
- DRW: enabled from epoch 5
- Early stopping metric: `balanced = 0.5 * qwk_mean + 0.5 * macro_recall`
- Early stopping patience: 25

TensorBoard event folders were copied from `runs/stage2_cv`. Because the live run directory was reused during development, the authoritative baseline numbers are the CSV/JSON files under `checkpoints/stage2_cv` in this archive.

## Main Results

5-fold mean metrics:

| Task | QWK | Accuracy | Precision macro |
| --- | ---: | ---: | ---: |
| DR | 0.9096 | 0.8083 | 0.7605 |
| ME | 0.8727 | 0.9358 | 0.8225 |

Training summary:

| Metric | Mean | Std |
| --- | ---: | ---: |
| qwk_dr | 0.9096 | 0.0146 |
| qwk_me | 0.8727 | 0.0171 |
| qwk_mean | 0.8912 | 0.0092 |
| macro_recall_dr | 0.7342 | 0.0344 |
| macro_recall_me | 0.8075 | 0.0648 |
| macro_recall | 0.7709 | 0.0244 |
| balanced | 0.8310 | 0.0098 |

## Held-Out Pooled Confusion Matrices

Rows are ground truth, columns are predictions.

DR:

```text
          pred0  pred1  pred2  pred3
true0      515     27      2      2
true1       61     70     21      1
true2       17     30    174     26
true3        2      0     41    211
```

DR per-class recall:

```text
DR0: 0.9432
DR1: 0.4575
DR2: 0.7045
DR3: 0.8307
```

ME:

```text
          pred0  pred1  pred2
true0      952     14      8
true1       19     45     11
true2       13     12    126
```

ME per-class recall:

```text
ME0: 0.9774
ME1: 0.6000
ME2: 0.8344
```

## Ablation Plan

Run these in separate output/log roots. Do not overwrite this archive.

1. DRW timing

   Compare `drw_defer_epoch=5` against `drw_defer_epoch=0`.

   Purpose: determine whether class reweighting should be active from the first epoch or after short warmup. Primary watch items are DR1, DR2, ME1 recall and `balanced`.

2. Early stopping metric

   Compare `early_stop_metric=balanced` against `early_stop_metric=qwk_mean`.

   Purpose: quantify how much the balanced criterion protects minority classes. Expect qwk_mean to favor majority-class stability; check whether macro recall drops.

3. Head/loss variant

   Compare current `head=corn` against `head=softmax` with LDAM+DRW.

   Purpose: test whether ordinal modeling or explicit LDAM margins are more useful for this dataset. Keep all other settings fixed.

4. Optional patience check

   Compare `early_stop_patience=25` against `35` only if some fold still appears undertrained.

   Purpose: verify the model is not stopping before minority-class recall peaks. This is lower priority because the current folds no longer stop extremely early.

Recommended ablation naming:

```text
checkpoints/stage2_cv_ablate_drw0
runs/stage2_cv_ablate_drw0

checkpoints/stage2_cv_ablate_qwkmean
runs/stage2_cv_ablate_qwkmean

checkpoints/stage2_cv_ablate_softmax_ldam
runs/stage2_cv_ablate_softmax_ldam
```

## Key Terms

- 5-fold cross-validation: Split data into 5 folds. Train on 4 folds and validate on the remaining fold, repeated 5 times. The final result summarizes all held-out folds.
- Held-out fold: The validation fold not used for training in that run. Metrics here are not training-set metrics.
- Patient-level split: Images from the same patient must stay in the same fold to avoid leakage.
- QWK: Quadratic Weighted Kappa. Agreement metric for ordinal labels; larger mistakes are penalized more than nearby mistakes.
- Recall: For a class, `true positives / all ground-truth samples of that class`. It answers how many samples of that class were recovered.
- Macro recall: Simple average of per-class recall. It gives rare classes the same weight as common classes.
- Precision macro: Simple average of per-class precision. It answers whether predictions assigned to each class are reliable, averaged equally over classes.
- DRW: Deferred Re-Weighting. Class weights are disabled before a chosen epoch and enabled later, so early representation learning is not dominated by noisy minority weighting.
- LDAM: Label-Distribution-Aware Margin loss. Adds larger margins for minority classes, usually used with softmax heads.
- CORN: Conditional Ordinal Regression for Neural networks. An ordinal head that predicts ordered thresholds instead of independent flat classes.
- Balanced early-stop metric: Current selection metric, `0.5 * qwk_mean + 0.5 * macro_recall`, designed to keep ordinal agreement while protecting minority-class recall.
