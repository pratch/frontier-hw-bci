# MI-EEG Classification: MixNet Baseline Guide & Reference

This repository contains the Motor Imagery EEG (MI-EEG) classification benchmark and baseline reproduction using **MixNet** on the **BCI Competition IV-2a (BCIC2a)** dataset.

This guide provides:
1. **Universal Environment Setup** (portable for any machine with Python and an NVIDIA GPU).
2. **Homework Roadmap & Implementation Checklist** (files to edit, commands to run, and benchmark workflow).
3. **Step-by-Step Baseline Reproduction** (dataset download, preprocessing, training, and benchmarking).
4. **Dataset Overview & Evaluation Protocol** (trial timing, channels, 5-fold cross-validation, and test rules).
5. **Deconstruction of the MixNet Pipeline Stages** (signal processing, network architecture, multi-task losses, and adaptive gradient blending).
6. **Configurable Parameters & Hyperparameters Reference Table**.

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

## 2. Homework Roadmap & Implementation Checklist

The objective is to design your own MI-EEG pipeline (**HWNet**), evaluate it across all 9 subjects on BCIC2a using 5-fold cross-validation, and statistically compare it against **MixNet**.

### Grading Breakdown
* **Part 1: Preprocessing & Feature Extraction** (30%)
* **Part 2: Neural Network Model Design** (30%)
* **Part 3: Report & Statistical Analysis** (40% — 15-minute presentation)

### Implementation Checklist

- [ ] **Phase 1: Environment & Baseline Reproduction**
  - [x] Configure Conda environment (`mixnet` with Python 3.8.10 & TF-GPU 2.7.0).
  - [ ] Download raw dataset: `python download_datasets.py --dataset BCIC2a`
  - [ ] Run baseline preprocessing: `python prep_spectral_spatial_signals.py --dataset BCIC2a --setting dependent`
  - [ ] Run MixNet on Subject 1 smoke test: `python run_MixNet.py ... --subjects 1`
  - [ ] Run MixNet across all 9 subjects: `python run_MixNet.py ...`
  - [ ] Generate baseline summary: `python benchmark.py`

- [ ] **Phase 2: Part 1 — Design Preprocessing & Feature Extraction (`HW_prep`)**
  - **Our pipeline, output shape and settings**: [`docs/hw_preprocessing.md`](docs/hw_preprocessing.md)
  - **File to Edit**: [`MixNet-homework/mixnet/preprocessing/BCIC2a/HW_prep.py`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/MixNet-homework/mixnet/preprocessing/BCIC2a/HW_prep.py)
    - Implement `fit_transform(X, y, pick_smp_freq, **kwargs)`: Learn data-driven parameters (e.g. CSP filters, scalers, whitening) on the **TRAINING fold only**, and return `(X_out, state)`.
    - Implement `transform(X, pick_smp_freq, state, **kwargs)`: Apply the learned state to validation and test folds without refitting.
    - **The Golden Rule**: Strictly avoid data leakage. No parameters may be fitted on validation folds or Session E.
  - **Script to Run**:
    ```bash
    cd MixNet-homework/experiments
    python prep_HW.py
    ```
  - **Output Path**: `MixNet-homework/experiments/datasets/BCIC2a/HW_prep/2_class/subject_dependent/` (30 `.npy` files per subject).

- [x] **Phase 3: Part 2 — Design Your Neural Network Architecture (`HWNet`)**
  - **Files Edited**:
    1. [`MixNet-homework/mixnet/models/HWNet.py`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/MixNet-homework/mixnet/models/HWNet.py): Implemented compact temporal CNN with ~5.4k parameters.
    2. [`MixNet-homework/experiments/configs/HWNet.py`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/MixNet-homework/experiments/configs/HWNet.py): Configured `NTCD` layout and training parameters.
  - **Scripts Executed**:
    ```bash
    cd MixNet-homework/experiments
    python run_HWNet.py --GPU 0
    ```
  - **Output Path**: `MixNet-homework/experiments/logs/HWNet/subject_dependent_2_classes_BCIC2a/` (contains all 9 subjects x 5 folds results).

- [x] **Phase 4: Part 3 — Benchmark & Compare Against MixNet Baseline**
  - **Script Executed**:
    ```bash
    cd MixNet-homework/experiments
    python benchmark.py --baseline MixNet --candidate HWNet
    ```
  - **Result**: HWNet achieved **77.13%** mean test accuracy vs. MixNet's **76.53%** (+0.60% accuracy, +0.40% macro F1).
  - **Outputs Produced**: `benchmark_per_subject.csv` and `benchmark_summary.csv`.

- [x] **Phase 5: Report & Presentation Preparation (15-min Presentation)**
  - Initial report draft created at [`docs/homework_report.md`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/homework_report.md).
  - Per-subject neurophysiological analysis and discussion documented.

> **Documentation References:**
> * HW Report Draft: [`docs/homework_report.md`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/homework_report.md)
> * Our HW preprocessing (Part 1): [`docs/hw_preprocessing.md`](docs/hw_preprocessing.md)
> * Concise MixNet paper TL;DR: [`docs/paper/mixnet_paper_summary.md`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/paper/mixnet_paper_summary.md)
> * Course Assignment Overview: [`docs/HWNet_Homework_Overview.pdf`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/HWNet_Homework_Overview.pdf)
> * MixNet Full Paper: [`docs/paper/MixNet_Paper.pdf`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/paper/MixNet_Paper.pdf)

---

## 3. Quickstart: Reproduce the Baseline

All experiment commands are executed from `MixNet-homework/experiments/`:

```bash
cd MixNet-homework/experiments
```

### Step 1: Download the BCIC2a Dataset
```bash
python download_datasets.py --dataset BCIC2a
```
* **Output Location**: `MixNet-homework/experiments/datasets/BCIC2a/raw/`
* **What is generated**:
  - 18 MATLAB `.mat` files (~15 MB each):
    - `A01T.mat` to `A09T.mat` (Session T: training and validation data for Subjects 1–9)
    - `A01E.mat` to `A09E.mat` (Session E: held-out evaluation data for Subjects 1–9)
  - Each file contains 22-channel continuous EEG signals along with event marker codes.

### Step 2: Run Baseline Preprocessing
```bash
python prep_spectral_spatial_signals.py --dataset BCIC2a --setting dependent
```
* **Output Location**: `MixNet-homework/experiments/datasets/BCIC2a/spectral_spatial_signals/2_class/2_csp_components/subject_dependent/`
* **What is generated**:
  - Total of **270 `.npy` files** (30 files per subject across 9 subjects).
  - For each subject (e.g., `S001`), the dataset is split into 5 stratified cross-validation folds:
    - `S001_fold01_X_train.npy`, `S001_fold01_y_train.npy` (4 folds of Session T used for training)
    - `S001_fold01_X_val.npy`, `S001_fold01_y_val.npy` (1 fold of Session T used for model checkpoint selection)
    - `S001_fold01_X_test.npy`, `S001_fold01_y_test.npy` (Untouched Session E test set)
    - Repeated for `fold02` through `fold05`.
  - **Data Shape**:
    - `X`: `(N_trials, 1, 400, 18)` — 400 time steps (4 seconds at 100 Hz), 18 channels (9 frequency sub-bands $\times$ 2 CSP components).
    - `y`: `(N_trials,)` — binary class labels (`0` = Left Hand, `1` = Right Hand).

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

* **Output Location**:
  `MixNet-homework/experiments/logs/MixNet/subject_dependent_2_classes_BCIC2a_HistoricalTangentSlope_csp_components_2_margin_1.0_latent_dim_18_warmup_7/`
* **What is generated per subject (e.g., `S001`)**:
  - **`S001_all_results.csv`**: Evaluation table summarizing test performance on held-out Session E across all 5 folds:
    - `test_acc`: Test accuracy on Session E (Subject 1 baseline reaches ~85.5% mean test accuracy).
    - `f1-score`: Macro F1-score across folds.
    - `test_loss`, `test_mse_loss`, `test_triplet_loss`, `test_crossentropy_loss`: Total and component test losses.
    - `w_mse_loss`, `w_triplet_loss`, `w_crossentropy_loss`: Final loss weights assigned by Adaptive Gradient Blending.
    - `prediction_time`, `memory_usage`: Inference latency per sample and memory footprint.
  - **`S001_fold01_out_log.csv` to `S001_fold05_out_log.csv`**: Epoch-by-epoch loss and accuracy curves (train and validation) for all training epochs.
  - **`S001_fold01_time_log.csv` to `S001_fold05_time_log.csv`**: Wall-clock training time recorded per epoch.
  - **`S001_prediction_results.npy`**: Saved raw model prediction probabilities and ground-truth labels on the test set.

### Step 4: Summarize Benchmark Results
```bash
python benchmark.py
```
* **Output Location**: `MixNet-homework/experiments/`
* **What is generated**:
  - **`benchmark_per_subject.csv`**: Aggregated table showing the 5-fold mean test accuracy and F1-score for each subject individually (`S001` through `S009`).
  - **`benchmark_summary.csv`**: Overall benchmark score summarizing mean performance and standard deviation across all subjects.
  - **Console Markdown Table**: Prints a paste-ready Markdown table comparing runs directly to stdout.

---

## 4. Understanding the BCIC2a Dataset

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

## 5. Deconstruction of the MixNet Baseline Stages

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

#### 1. Sub-Band Decomposition
Decomposes the 20 EEG channels into 9 adjacent frequency sub-bands:
`[4, 8]`, `[8, 12]`, `[12, 16]`, `[16, 20]`, `[20, 24]`, `[24, 28]`, `[28, 32]`, `[32, 36]`, `[36, 40]` Hz
using a 5th-order Butterworth bandpass filter.

#### 2. Common Spatial Pattern (CSP)
For each sub-band $b$, computes normalized average spatial covariance matrices:

$$
\Sigma_1 = \frac{1}{N_1} \sum \frac{X X^T}{\text{Tr}(X X^T)}, \quad \Sigma_2 = \frac{1}{N_2} \sum \frac{X X^T}{\text{Tr}(X X^T)}
$$

Solves the generalized eigenvalue problem:

$$
\Sigma_1 W = \lambda (\Sigma_1 + \Sigma_2) W
$$

Picks $n_{\text{components}} = 2$ spatial filters (extremal eigenvalues) per band.

#### 3. Spectral-Spatial Signal Formation
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

$$
\mathcal{L}_{\text{total}} = w_{\text{mse}} \mathcal{L}_{\text{mse}}(x, x_r) + w_{\text{triplet}} \mathcal{L}_{\text{triplet}}(y, z) + w_{\text{ce}} \mathcal{L}_{\text{ce}}(y, \hat{y})
$$

1. **$\mathcal{L}_{\text{mse}}$ (Reconstruction)**: Mean squared error forcing latent vector $z$ to retain signal information.
2. **$\mathcal{L}_{\text{triplet}}$ (Metric Learning)**: Semi-hard triplet loss with `margin=1.0`, clustering same-class latent vectors and separating opposite-class vectors.
3. **$\mathcal{L}_{\text{ce}}$ (Classification)**: Sparse categorical cross-entropy predicting Left vs. Right hand.

### Stage 4: Adaptive Gradient Blending (`HistoricalTangentSlope`)
- Instead of keeping loss weights fixed, MixNet dynamically adjusts $w_{\text{mse}}$, $w_{\text{triplet}}$, and $w_{\text{ce}}$ based on the validation curve slope of each loss over recent epochs.
- A **warmup period** of 7 epochs trains with initial weights before adaptive blending activates.

---

## 6. Configurable Parameters & Hyperparameters Reference

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

## 7. Repository Layout

```text
frontier-hw-bci/
├── docs/                                          # Course & reference PDFs and guides
│   ├── final_slides/                              # Final combined presentation (Part 1 + Part 2)
│   ├── milestone_slides/                          # Earlier decks: HWNet_Presentation, prep_slides/ (Part 1 source)
│   ├── paper/                                     # MixNet paper, supplementary & TL;DR summary
│   ├── homework_report.md                         # Homework report
│   ├── hyperparameter_tuning_report.md            # HWNet tuning report
│   ├── hw_preprocessing.md                        # Our Part 1 pipeline: usage & output shape
│   └── HWNet_Homework_Overview.pdf                # Homework overview slides
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
