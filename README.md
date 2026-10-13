# MI-EEG Classification: MixNet Baseline Guide & Reference

This repository contains the Motor Imagery EEG (MI-EEG) classification benchmark and baseline reproduction using **MixNet** on the **BCI Competition IV-2a (BCIC2a)** dataset.

This guide provides:
1. **Universal Environment Setup** (portable for any machine with Python and an NVIDIA GPU).
2. **Step-by-Step Baseline Reproduction** (dataset download, preprocessing, training, and benchmarking).
3. **Dataset Overview & Evaluation Protocol** (trial timing, channels, 5-fold cross-validation, and test rules).
4. **Deconstruction of the MixNet Pipeline Stages** (signal processing, network architecture, multi-task losses, and adaptive gradient blending).
5. **Configurable Parameters & Hyperparameters Reference Table**.

---

## 1. Environment Setup (Any GPU Machine)

The codebase requires **Python 3.8** and **TensorFlow 2.7.0**.

### 1.1 Create Conda Environment
```bash
conda create -n mixnet python=3.8.10 -y
conda activate mixnet
```

### 1.2 Install Dependencies
From the repository root:
```bash
cd MixNet-homework

# Install base dependencies
pip install -r requirements.txt

# Install TensorFlow GPU and compatible protobuf
pip install tensorflow-gpu==2.7.0 "protobuf<=3.20.3"

# Install mixnet package in editable mode
pip install -e .
```

Verify the environment:
```bash
python -c "import mixnet; print('MixNet package:', mixnet.__file__); import tensorflow as tf; print('TensorFlow:', tf.__version__, 'GPUs:', tf.config.list_physical_devices('GPU'))"
```

---

## 2. Quickstart: Reproduce the Baseline

All experiment commands are executed from `MixNet-homework/experiments/`:

```bash
cd MixNet-homework/experiments
```

### Step 1: Download the BCIC2a Dataset
```bash
python download_datasets.py --dataset BCIC2a
```
* Automatically downloads all 18 files (`A01T.mat`–`A09T.mat` and `A01E.mat`–`A09E.mat`) into `MixNet-homework/experiments/datasets/BCIC2a/raw/`.

### Step 2: Run Baseline Preprocessing
```bash
python prep_spectral_spatial_signals.py --dataset BCIC2a --setting dependent
```
* Decomposes signals across 9 sub-bands and fits Common Spatial Pattern (CSP) filters.
* Generates 30 `.npy` files per subject across 5 folds (270 files total) in `MixNet-homework/experiments/datasets/BCIC2a/spectral_spatial_signals/2_class/subject_dependent/`.

### Step 3: Train MixNet (Subject-Dependent Baseline)
```bash
python run_MixNet.py \
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
    --warmup 7
```
*(Tip: To test on a single subject first, append `--subjects 1` to the command above).*

### Step 4: Summarize Benchmark Results
```bash
python benchmark.py
```
* Aggregates fold performance for each subject and generates:
  - `benchmark_per_subject.csv` (per-subject mean test accuracy & F1-score)
  - `benchmark_summary.csv` (overall dataset mean ± standard deviation)

---

## 3. Understanding the BCIC2a Dataset

The **BCI Competition IV-2a** (`BCIC2a`) is a standard benchmark for 2-class motor imagery (Left Hand vs. Right Hand).

### Key Dataset Properties
| Property | Value | Description |
|---|---|---|
| **Subjects** | 9 (`A01`–`A09`) | Evaluated individually in a subject-dependent setting. |
| **Classes** | 2 | Class `0`: Left Hand, Class `1`: Right Hand. |
| **Sessions** | `T` and `E` | `T` = Training/Validation session; `E` = Untouched Test session recorded on a different day. |
| **Trials per session** | 144 | 72 left hand trials, 72 right hand trials per session. |
| **Channels** | 20 electrodes | 20 motor-cortex electrodes selected out of 22 original channels. |
| **Sampling Rate** | 100 Hz | Downsampled from 250 Hz (1 sample = 10 ms). |
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
  Fixation Cross  Visual Cue (Left / Right)     Motor Imagery Execution
  (Preparation)  (Arrow prompt)                (Feature Window: 400 samples)
```

### Evaluation Protocol & The Golden Rule
* **Session T (Train / Validation)**: Divided using **5-fold stratified cross-validation** (4 folds train, 1 fold validation per fold).
* **Session E (Test)**: **Untouched test set**. Never used for training or hyperparameter selection. At the end of each fold, the model checkpoint with the highest validation score is evaluated on Session E.

---

## 4. Deconstruction of the MixNet Baseline Stages

MixNet combines classical signal processing (Filter Bank CSP) with a multi-task deep representation learning network.

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
   Decomposes the 20 EEG channels into 9 adjacent frequency sub-bands:
   $$[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]\text{ Hz}$$
   using a 5th-order Butterworth bandpass filter.
2. **Common Spatial Pattern (CSP)**:
   For each sub-band $b$, computes normalized average spatial covariance matrices:
   $$\Sigma_1 = \frac{1}{N_1} \sum \frac{X X^T}{\text{Tr}(X X^T)}, \quad \Sigma_2 = \frac{1}{N_2} \sum \frac{X X^T}{\text{Tr}(X X^T)}$$
   Solves the generalized eigenvalue problem:
   $$\Sigma_1 W = \lambda (\Sigma_1 + \Sigma_2) W$$
   Picks $n_{\text{components}} = 2$ spatial filters (extremal eigenvalues) per band.
3. **Spectral-Spatial Signal Formation**:
   Applies spatial projection: $S_b = W_b^T X_b$.
   Concatenates across all 9 bands $\rightarrow 9 \times 2 = 18$ channels.
   Final input tensor shape per trial: `(1, 400, 18)` (height=1, time=400, channels=18).

### Stage 2: Neural Network Architecture (`MixNet-homework/mixnet/models/MixNet.py`)
* **Encoder (`encoder`)**:
  1. `Conv2D(filters=18, kernel_size=(1, 64), activation='elu', padding='same')`:
     - 1D temporal convolution across time ($64\text{ samples} = 640\text{ ms}$ temporal filter).
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

### Stage 3: Multi-Task Losses (`MixNet-homework/mixnet/loss.py`)
MixNet trains jointly with three losses:
$$\mathcal{L}_{\text{total}} = w_{\text{mse}} \mathcal{L}_{\text{mse}}(x, x_r) + w_{\text{triplet}} \mathcal{L}_{\text{triplet}}(y, z) + w_{\text{ce}} \mathcal{L}_{\text{ce}}(y, \hat{y})$$
1. **$\mathcal{L}_{\text{mse}}$ (Reconstruction)**: Mean squared error forcing latent vector $z$ to retain signal information.
2. **$\mathcal{L}_{\text{triplet}}$ (Metric Learning)**: Semi-hard triplet loss with `margin=1.0`, clustering same-class latent vectors and separating opposite-class vectors.
3. **$\mathcal{L}_{\text{ce}}$ (Classification)**: Sparse categorical cross-entropy predicting Left vs. Right hand.

### Stage 4: Adaptive Gradient Blending (`HistoricalTangentSlope`)
- Instead of keeping loss weights fixed, MixNet dynamically adjusts $w_{\text{mse}}$, $w_{\text{triplet}}$, and $w_{\text{ce}}$ based on the validation curve slope of each loss over recent epochs.
- A **warmup period** of 7 epochs trains with initial weights before adaptive blending activates.

---

## 5. Configurable Parameters & Hyperparameters Reference

### Command-Line Arguments (`MixNet-homework/experiments/run_MixNet.py`)
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

### Model & Training Hyperparameters (`MixNet-homework/experiments/configs/MixNet.py`)
| Parameter | Default | Description |
|---|---|---|
| `input_shape` | `(1, 400, 18)` | Shape of each input trial (excluding batch dimension). |
| `latent_dim` | `18` | Dimension of latent bottleneck vector $z$. |
| `batch_size` | `10` | Mini-batch size during training. |
| `lr` | `1e-3` | Initial learning rate for Adam optimizer. |
| `epochs` | `200` | Maximum number of training epochs per fold. |
| `es_patience` | `20` | Early stopping patience (stops if validation loss stalls). |
| `loss_weights` | `[1.0, 1.0, 1.0]` | Initial weights $[\beta_{\text{mse}}, \beta_{\text{triplet}}, \beta_{\text{ce}}]$. |

---

## 6. Repository Layout

```text
frontier-hw-bci/
├── docs/                                          # Course & reference PDFs
│   ├── HWNet_Homework_Overview.pdf                # Homework overview slides
│   ├── MixNet_Paper.pdf                           # MixNet main IEEE/arXiv paper
│   └── Supplementary_Materials_for_MixNet_...pdf  # MixNet supplementary details
├── MixNet-homework/                               # Original homework starter repository
│   ├── experiments/                               # Experiment runners, configs & datasets
│   ├── mixnet/                                    # MixNet model & preprocessing library
│   ├── imgs/                                      # Images from original repo
│   ├── INSTRUCTION.md                             # Official homework instruction markdown
│   ├── LICENSE                                    # Apache 2.0 license
│   ├── README.md                                  # Original MixNet README
│   ├── requirements.txt                           # Dependency specifications
│   └── setup.py                                   # Package setup
├── README.md                                      # This guide
└── .gitignore
```
