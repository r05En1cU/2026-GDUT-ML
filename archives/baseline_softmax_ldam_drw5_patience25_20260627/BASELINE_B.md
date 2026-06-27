# Baseline B: Softmax + LDAM + DRW

This archive freezes the category-sensitive Stage-2 baseline.

## Configuration

```text
backbone: ConvNeXt-Tiny
attention: cross
head: softmax
loss: LDAM + DRW
drw_defer_epoch: 5
early_stop_metric: balanced
balanced: 0.5 * qwk_mean + 0.5 * macro_recall
early_stop_patience: 25
evaluation: 5-fold held-out
```

Config snapshot:

```text
archives/baseline_softmax_ldam_drw5_patience25_20260627/stage2_softmax_ldam.yaml
```

Source checkpoint directory:

```text
checkpoints/stage2_cv_ablate_softmax_ldam
```

## 5-Fold Summary

Mean across folds:

```text
DR QWK:          0.9028 +/- 0.0159
ME QWK:          0.8386 +/- 0.0387
QWK mean:        0.8707 +/- 0.0183
macro recall:    0.7851 +/- 0.0405
balanced:        0.8279 +/- 0.0253
```

## Pooled Held-Out Results

DR:

```text
n = 1200
QWK = 0.9026
recall = [0.9066, 0.5163, 0.6154, 0.8583]

confusion matrix:
[[495, 49,   0,   2],
 [ 47, 79,  25,   2],
 [ 21, 31, 152,  43],
 [  1,  0,  35, 218]]
```

ME:

```text
n = 1200
QWK = 0.8388
recall = [0.9497, 0.7867, 0.7748]

confusion matrix:
[[925, 37,  12],
 [  8, 59,   8],
 [ 14, 20, 117]]
```

## Positioning

Baseline B is not the primary stability baseline. It is the category-sensitive baseline.

Compared with the CORN baseline:

```text
Strength:
- Better DR class-1 recall.
- Better ME class-1 recall.
- More direct optimization of class boundaries through LDAM.

Weakness:
- Lower ME QWK.
- Weaker ordinal consistency.
- DR class-2 recall is lower than CORN.
```

This baseline is useful because it exposes the trade-off that the next model should improve:

```text
Keep the middle-class recall gain of softmax+LDAM,
while recovering the ordinal stability of CORN.
```

## Post-Hoc Projection Diagnostics

Two projection diagnostics are archived:

```text
projection_eval_fixed_t05.json
projection_search_eval.json
```

Fixed threshold `0.5` failed because softmax class probabilities are not ordinal tail probabilities. Cross-fold threshold search improved the fixed projection failure mode, but still underperformed raw softmax and CORN.

Conclusion:

```text
Do not use softmax -> CORN projection as the main path.
Use this baseline as a reference for category-sensitive recall.
```
