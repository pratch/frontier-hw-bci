# Week 1 Guide: BCIC2a Dataset, MixNet Baseline Reproduction & Architecture Deconstruction

This document is focused on the goals for **Week 1**:
1. **Understanding the BCI Competition IV-2a (BCIC2a) Dataset** (trial structure, channels, sessions, and evaluation rules).
2. **Reproducing the MixNet Baseline** step-by-step on `v21`.
3. **Deconstructing All Stages of the Baseline** (preprocessing, neural network architecture, loss functions, and optimization dynamics).
4. **Reference Table of Configurable Hyperparameters and Arguments**.

*(Note: Ideas for custom pipelines and future assignment tasks have been separated into `todo/future_ideas_and_improvements.md` for later reference).*

---

## 1. Understanding the BCIC2a Dataset

The **BCI Competition IV-2a** (`BCIC2a`) is a benchmark dataset for 2-class and 4-class motor imagery (MI). In this assignment, we use the **2-class motor imagery task (Left Hand vs. Right Hand)** in a **subject-dependent** setting.

### Key Dataset Properties
| Property | Value | Description |
|---|---|---|
| **Subjects** | 9 (`A01`–`A09`) | Evaluated individually (subject-dependent models). |
| **Classes** | 2 | Class `0`: Left Hand, Class `1`: Right Hand. |
| **Sessions** | `T` & `E` | `T` = Training/Validation session; `E` = Untouched Test session recorded on a different day. |
| **Trials per session** | 144 | 72 left hand trials, 72 right hand trials per session. |
| **Channels** | 20 electrodes | 20 motor-cortex electrodes selected out of 22 original channels. |
| **Sampling Rate** | 100 Hz | Downsampled from the original 250 Hz (1 sample = 10 ms). |
| **MI Trial Window** | 4.0 seconds | Samples `t = 2.0 s` to `6.0 s` after cue $\rightarrow$ **400 time steps**. |

### 20 Selected Motor Electrodes
The selected electrodes cover the bilateral sensorimotor cortex:
```text
Fz, FC3, FC1, FCz, FC2, FC4, C5, C3, C1, Cz, C2, C4, C6, CP3, CP1, CPz, CP2, CP4, P1, P2
```
Electrodes $C_3$ (left hemisphere) and $C_4$ (right hemisphere) are the primary sites where contralateral Event-Related Desynchronization (ERD) occurs during right-hand and left-hand imagery.

### Trial Timing & Protocol
```text
0.0 s          2.0 s                                         6.0 s
  |--------------|---------------------------------------------|
  Fixation Cross  Cue Arrow (Left / Right)     Motor Imagery Execution
  (Prepare)      (Visual stimulus)            (Feature Extraction Window: 400 samples)
```

### The Golden Evaluation Rule
* **Session T (Train / Validation)**: Divided using **5-fold stratified cross-validation** (4 folds train, 1 fold validation per fold).
* **Session E (Test)**: **Untouched test set**. Never used for model training or hyperparameter selection. At the end of each fold, the model checkpoint with the best validation performance is evaluated on Session E.

---

## 2. Reproducing the MixNet Baseline on `v21`

A pre-configured Conda environment and GPU runner script are set up on `v21`. All commands should be run from `experiments/`:

### Step 1: Download the Dataset
```bash
# SSH into v21 and run download script
ssh v21 'cd /ist-nas/users/pratchp/homework/frontier-hw-bci/experiments && ../scripts/run_v21.sh python download_datasets.py --dataset BCIC2a'
```
* Downloads all 18 raw files (`A01T.mat`–`A09T.mat` and `A01E.mat`–`A09E.mat`) into `experiments/datasets/BCIC2a/raw/`.

### Step 2: Run Baseline Preprocessing
```bash
ssh v21 'cd /ist-nas/users/pratchp/homework/frontier-hw-bci/experiments && ../scripts/run_v21.sh python prep_spectral_spatial_signals.py --dataset BCIC2a --setting dependent'
```
* Generates spectral-spatial signals using sub-band Common Spatial Patterns (CSP).
* Produces 30 `.npy` files per subject across 5 folds (270 `.npy` files total) in:
  `experiments/datasets/BCIC2a/spectral_spatial_signals/2_class/subject_dependent/`.

### Step 3: Train MixNet (Subject-Dependent Baseline)
```bash
ssh v21 'cd /ist-nas/users/pratchp/homework/frontier-hw-bci/experiments && ../scripts/run_v21.sh python run_MixNet.py \
    --model_name MixNet \
    --dataset BCIC2a \
    --train_type subject_dependent \
    --data_type spectral_spatial_signals \
    --adaptive_gradient True \
    --policy HistoricalTangentSlope \
    --log_dir logs \
    --num_class 2 \
    --GPU 0 \
    --margin 1.0 \
    --n_component 2 \
    --warmup 7'
```
*(To test a single subject first, append `--subjects 1`).*

### Step 4: Summarize Baseline Benchmark
```bash
ssh v21 'cd /ist-nas/users/pratchp/homework/frontier-hw-bci/experiments && ../scripts/run_v21.sh python benchmark.py'
```
* Aggregates 5 folds for each subject and prints mean accuracy, F1-scores, and standard deviations into:
  - `benchmark_per_subject.csv`
  - `benchmark_summary.csv`

---

## 3. Deconstruction of the MixNet Baseline Stages

MixNet joins classical BCI feature extraction (Filter Bank CSP) with a modern multi-task autoencoder network.

```
[Raw EEG: 20 ch, 400 samples]
               │
               ▼
┌────────────────────────────────────────┐
│  STAGE 1: Spectral-Spatial Filtering   │
│  - 9 frequency sub-bands               │
│  - Per-band CSP spatial projection     │
│  - Output: (1, 400, 18) continuous     │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  STAGE 2: Multi-Task Architecture      │
│  - Encoder: 1D Temporal Convolutions   │
│  - Bottleneck: Latent space z (dim=18) │
│  - Decoder: Conv2DTranspose (xr)       │
│  - Classifier: Softmax Dense (y)       │
└──────────────────┬─────────────────────┘
                   │
                   ▼
┌────────────────────────────────────────┐
│  STAGE 3 & 4: Multi-Task Losses & AGB  │
│  - Loss: MSE (xr) + Triplet (z) + CE(y)│
│  - Adaptive Gradient Blending (HTS)    │
└────────────────────────────────────────┘
```

### Stage 1: Preprocessing & CSP Signal Generation (`prep_spectral_spatial_signals.py`)
1. **Sub-Band Decomposition**:
   Decomposes the 20 EEG channels into 9 overlapping/adjacent frequency sub-bands:
   $$[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]\text{ Hz}$$
   using a 5th-order Butterworth bandpass filter.
2. **Common Spatial Pattern (CSP)**:
   For each sub-band $b$, computes the normalized average spatial covariance for each class:
   $$\Sigma_1 = \frac{1}{N_1} \sum \frac{X X^T}{\text{Tr}(X X^T)}, \quad \Sigma_2 = \frac{1}{N_2} \sum \frac{X X^T}{\text{Tr}(X X^T)}$$
   Solves the generalized eigenvalue problem:
   $$\Sigma_1 W = \lambda (\Sigma_1 + \Sigma_2) W$$
   Picks $n_{\text{components}} = 2$ spatial filters (extremal eigenvalues) per band.
3. **Spectral-Spatial Signal Formation**:
   Applies the spatial filters: $S_b = W_b^T X_b$.
   Concatenates across the 9 bands $\rightarrow 9 \times 2 = 18$ channels.
   Final input tensor shape per trial: `(1, 400, 18)` (height=1, time=400, channels=18).

### Stage 2: Neural Network Architecture (`mixnet/models/MixNet.py`)
* **Encoder (`encoder`)**:
  1. `Conv2D(filters=18, kernel_size=(1, 64), activation='elu', padding='same')`:
     - 1D temporal convolution across time ($64\text{ samples} = 640\text{ ms}$ temporal receptive field).
     - Weight constraint: `max_norm(2.0)`.
  2. `BatchNormalization` + `AveragePooling2D((1, 1))`.
  3. `Conv2D(filters=9, kernel_size=(1, 32), activation='elu', padding='same')`:
     - Second temporal convolution ($32\text{ samples} = 320\text{ ms}$).
  4. `BatchNormalization` + `AveragePooling2D((1, 4))`:
     - Downsamples time from 400 to 100 samples ($4 \times$ reduction).
  5. `Flatten` $\rightarrow$ `Dense(LATENT_DIM=18, kernel_constraint=max_norm(0.5))` $\rightarrow$ latent representation $z$.
* **Decoder (`decoder`)**:
  - `Dense` expanding $z \rightarrow (1, 25, 9)$.
  - `Conv2DTranspose(strides=(1, 4))` reconstructing temporal dimension back to 100.
  - `Conv2DTranspose(strides=(1, 1))` reconstructing back to $(1, 400, 18)$ (`xr`).
* **Classifier**:
  - `Dense(num_class=2, activation='softmax', kernel_constraint=max_norm(0.5))` directly on $z$.

### Stage 3: Multi-Task Losses (`mixnet/loss.py`)
MixNet trains jointly with three losses:
$$\mathcal{L}_{\text{total}} = w_{\text{mse}} \mathcal{L}_{\text{mse}}(x, x_r) + w_{\text{triplet}} \mathcal{L}_{\text{triplet}}(y, z) + w_{\text{ce}} \mathcal{L}_{\text{ce}}(y, \hat{y})$$
1. **$\mathcal{L}_{\text{mse}}$ (Reconstruction)**: Mean squared error forcing $z$ to retain signal information.
2. **$\mathcal{L}_{\text{triplet}}$ (Metric Learning)**: Semi-hard triplet loss with `margin=1.0`, clustering same-class latent vectors and repelling opposite-class vectors.
3. **$\mathcal{L}_{\text{ce}}$ (Classification)**: Sparse categorical cross-entropy predicting Left vs. Right hand.

### Stage 4: Adaptive Gradient Blending (`HistoricalTangentSlope`)
- Instead of keeping loss weights fixed, MixNet dynamically adjusts $w_{\text{mse}}$, $w_{\text{triplet}}$, and $w_{\text{ce}}$ based on the slope of each loss's validation curve.
- A **warmup period** of 7 epochs trains with initial weights before adaptive blending activates.

---

## 4. Configurable Parameters & Hyperparameters

Below is a reference guide to all key configurable settings in the MixNet baseline:

### Command-Line Arguments (`experiments/run_MixNet.py`)
| Parameter | Default | Description |
|---|---|---|
| `--dataset` | `BCIC2a` | Dataset name (`BCIC2a`, `BCIC2b`, etc.). |
| `--train_type` | `subject_dependent` | Subject-dependent (individual models) or independent. |
| `--data_type` | `spectral_spatial_signals` | Preprocessed feature folder to load. |
| `--n_component` | `2` | Number of CSP spatial filters picked per frequency band ($2 \times 9 = 18$ channels). |
| `--margin` | `1.0` | Margin $\alpha$ for semi-hard triplet loss. |
| `--warmup` | `7` | Warmup epochs before adaptive gradient blending takes effect. |
| `--adaptive_gradient` | `True` | Whether to dynamically balance loss weights. |
| `--policy` | `HistoricalTangentSlope` | Adaptive gradient policy (`HistoricalTangentSlope` or `Fixed`). |
| `--subjects` | `None` (all 1–9) | Specific subject(s) to train, e.g. `--subjects 1`. |
| `--GPU` | `0` | GPU index to run on. |

### Model & Training Parameters (`experiments/configs/MixNet.py`)
| Parameter | Default | Description |
|---|---|---|
| `input_shape` | `(1, 400, 18)` | Shape of each input trial (excluding batch dimension). |
| `latent_dim` | `18` | Dimension of latent bottleneck vector $z$. |
| `batch_size` | `10` | Mini-batch size during training. |
| `lr` | `1e-3` | Initial learning rate for Adam optimizer. |
| `epochs` | `200` | Maximum number of training epochs per fold. |
| `es_patience` | `20` | Early stopping patience (stops if validation loss stalls). |
| `loss_weights` | `[1.0, 1.0, 1.0]` | Initial weights $[\beta_{\text{mse}}, \beta_{\text{triplet}}, \beta_{\text{ce}}]$. |
