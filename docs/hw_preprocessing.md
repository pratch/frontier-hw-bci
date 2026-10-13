# HW Preprocessing (Part 1)

Code: [`MixNet-homework/mixnet/preprocessing/BCIC2a/HW_prep.py`](../MixNet-homework/mixnet/preprocessing/BCIC2a/HW_prep.py)
Settings: [`MixNet-homework/experiments/prep_HW.py`](../MixNet-homework/experiments/prep_HW.py)

## Idea

Classic CSP gives one log-variance number per trial, so it loses *when* the ERD/ERS happens inside the 4 s window (lecture p.62).
We keep MixNet's filter bank + CSP, but compute log-variance in short sliding windows, so the model sees how band power changes over time.

## Pipeline

```
raw EEG (n, 20, 400)
  1. filter bank (9 bands, 4-40 Hz) + CSP per band, 4 components   -> (n, 36, 400)
  2. normalized log-variance in 0.5 s windows, 0.25 s step         -> (n, 36, 15)
  3. per-channel standardization (mean/std of the training fold)   -> (n, 36, 15)
```

| Step | Learned from data? | Lecture ref |
|---|---|---|
| 1. CSP filters | yes, training fold only | "Fit CSP: training data only" (p.60) |
| 2. Windowed log-variance | no | normalized log-variance (p.59) |
| 3. Standardization | yes, training fold only | "Training-set per-channel standardization" (p.31) |

`transform()` only reads the saved `state`, so validation and test data never refit anything.

## Output

Saved to `MixNet-homework/experiments/datasets/BCIC2a/HW_prep/2_class/subject_dependent/` (30 `.npy` files per subject, 270 total).

| File | Shape | Notes |
|---|---|---|
| `X_*` | `(n, 36, 15)` | `float32` |
| `y_*` | `(n,)` | `0` = left hand, `1` = right hand |

Axis 1 (36 features) is ordered **band first**: features 0-3 are the 4 CSP components of 4-8 Hz, 4-7 are 8-12 Hz, and so on up to 36-40 Hz.
Axis 2 (15) is time windows, from the start to the end of the 4 s MI window.

To get one band: `X[:, b*4:(b+1)*4, :]`. To get `(n, 9, 4, 15)`: `X.reshape(n, 9, 4, 15)`.

## How to run

```bash
cd MixNet-homework/experiments
python prep_HW.py                                    # about 1 min, all 9 subjects

cd ..
python -m mixnet.preprocessing.BCIC2a.HW_prep        # self-check, synthetic data, a few seconds
```

The self-check fails if shapes break, if the training fold is not standardized, or if any statistic leaks from other trials into a trial's features.

`datasets/` is gitignored, so everyone has to run `prep_HW.py` on their own machine.

## Settings (`prep_HW.py`)

| Setting | Default | Changes output shape? |
|---|---|---|
| `bands` | 9 bands, 4 Hz wide, 4-40 Hz (same as MixNet) | yes, axis 1 |
| `order` | `5` (Butterworth, same as MixNet) | no |
| `n_components` | `4` (2 from each end of CSP) | yes, axis 1 |
| `win_len` | `0.5` s | yes, axis 2 |
| `win_step` | `0.25` s | yes, axis 2 |
| `norm_logvar` | `True` | no |

Shape formula:

```
axis 1 = len(bands) * n_components
axis 2 = (400 - win_len*100) // (win_step*100) + 1
```

If you change a setting that changes the shape, tell whoever owns the model and update `input_shape` in `configs/HWNet.py`.

## For the model owner

Set these in [`experiments/configs/HWNet.py`](../MixNet-homework/experiments/configs/HWNet.py). `data_format` decides what shape the model receives (see the table at the top of that file).

| `data_format` | Model input shape | Good for |
|---|---|---|
| `None` | `(36, 15)` | MLP, or Conv1D/GRU after a `Permute` |
| `'NTCD'` | `(1, 15, 36)` | Conv2D with `channels_last`, same layout as MixNet `(1, 400, 18)` |
| `'NCTD'` | `(36, 15, 1)` | Conv2D with `channels_last`, features as image rows |

Use `channels_last` in the model. `channels_first` Conv2D does not run on CPU (including Mac).

## Quick sanity check (not a result)

Shrinkage-LDA on the **validation folds of session T** (session E not used), mean over 9 subjects:

| Variant | Val accuracy |
|---|---|
| Default (normalized, 0.5 s windows) | 84.7% |
| `norm_logvar=False` | 85.0% |
| One 4 s window (= classic CSP log-variance) | 82.0% |

This only shows the features carry class information and that time windows help a little.
Do **not** compare these numbers with MixNet's test accuracy on session E (76.77% in our run).

## Ablation ideas

Each one is a single setting change in `prep_HW.py`, then rerun prep and the model:

- `win_len=4.0, win_step=4.0`: no time windows (classic CSP log-variance)
- `norm_logvar=False`: plain log-variance
- `n_components=2`: same CSP size as MixNet
