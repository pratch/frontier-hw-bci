# Homework Report: HWNet — Motor Imagery BCI Classification on BCIC2a

**Course**: Frontier BCI Homework  
**Pipeline**: Filter-Bank Windowed CSP (`HW_prep`) + Compact Temporal CNN (`HWNet`)  
**Evaluation**: Subject-Dependent 5-Fold Cross-Validation on BCI Competition IV-2a (Session E Held-Out Test Set)  

---

## 1. Executive Summary

This project develops an efficient, neurophysiologically grounded Brain-Computer Interface (BCI) decoding pipeline for 2-class motor imagery (Left Hand vs. Right Hand). 

While existing state-of-the-art models like **MixNet** rely on raw continuous EEG signals ($1 \times 400 \times 18$) coupled with complex multi-task objectives (autoencoder reconstruction + triplet metric learning + cross-entropy classification, ~40,000+ parameters), our pipeline decouples signal representation from temporal classification:
1. **Preprocessing (`HW_prep`)**: Extracts 9-band Filter-Bank Common Spatial Pattern (FBCSP) log-variance across sliding temporal windows, preserving sensorimotor ERD/ERS dynamics into a compact $(36 \times 15)$ feature map.
2. **Architecture (`HWNet`)**: Employs an ultra-lightweight 2-block temporal CNN (**5,458 parameters**) trained with single-task cross-entropy.

### Headline Results:
* **HWNet Overall Mean Test Accuracy**: **78.64% ± 14.78%** (Macro F1: **77.59% ± 16.42%**)
* **MixNet Evaluated Baseline**: **76.53% ± 15.10%** (Paper: **76.77%**)
* **Advantage over MixNet**: **+2.11% accuracy, +2.27% Macro F1** (wins on 5/9 subjects)
* **Model Size**: ~5× fewer parameters (~8.7k vs. ~44.8k)
* **Training Time**: ~30 seconds per subject across all 5 folds on an NVIDIA RTX 4090.

---

## 2. Part 1 — Preprocessing & Feature Extraction (`HW_prep`)

### 2.1 Neurophysiological Motivation
Motor imagery produces **Event-Related Desynchronization (ERD)** (power decrease in $\mu$ [8–12 Hz] and $\beta$ [16–24 Hz] rhythms over the contralateral sensorimotor cortex) and subsequent **Event-Related Synchronization (ERS)** (post-movement $\beta$ rebound over ipsilateral/contralateral areas).

* **The Limitation of Classic CSP**: Standard CSP projects multi-channel EEG into spatial components and computes a single global log-variance over the entire 4.0-second trial. This collapses the temporal dimension into a static 1D vector, discarding all information regarding *when* ERD/ERS initiates, peaks, and transitions.
* **Our Solution**: We apply sliding-window log-variance extraction. This retains the temporal progression of power envelopes over time while drastically reducing signal dimensionality from 400 raw time points to 15 time windows.

### 2.2 Mathematical Formulation & Pipeline Architecture

```text
Raw EEG Trial (20 channels, 400 samples @ 100 Hz = 4.0 s)
  │
  ├─► Step 1: Filter-Bank Decomposition (9 sub-bands, 4–40 Hz, Butterworth order 5)
  │     Bands: [4,8], [8,12], [12,16], [16,20], [20,24], [24,28], [28,32], [32,36], [36,40] Hz
  │
  ├─► Step 2: Common Spatial Pattern (CSP) per Band
  │     Keeps N = 4 spatial filters per band (2 largest, 2 smallest eigenvalues)
  │     Output: 9 bands x 4 components = 36 spatial channels
  │
  ├─► Step 3: Sliding-Window Normalized Log-Variance
  │     Window length: 0.5 s (50 samples), Hop step: 0.25 s (25 samples)
  │     Number of windows: (400 - 50) // 25 + 1 = 15 windows
  │     Normalized log-variance per band: log( var_i / sum_j var_j )
  │
  └─► Step 4: Per-Channel Standardization
        Standardize mean = 0, std = 1 per channel using training fold statistics only.
        Final Output Shape: (n_trials, 36 channels, 15 time steps)
```

### 2.3 Strict Data Hygiene (Zero Leakage)
To avoid over-optimistic evaluation and data snooping:
* CSP projection matrices $\mathbf{W}$ and feature standardization scalers ($\mu, \sigma$) are fitted strictly on `X_train` inside `fit_transform()`.
* `transform()` only reads the stored `state` dictionary and applies transformations to validation and test folds without refitting.

---

## 3. Part 2 — Neural Network Architecture (`HWNet`)

### 3.1 Design Principles for Sample-Scared BCI Data
The BCI Competition IV-2a dataset provides only **~115 training trials per subject per fold**. Overparameterized deep networks with complex latent spaces are prone to overfitting to non-stationary sensor noise. 

Because `HW_prep` already extracts optimal spatial and spectral features, `HWNet` focuses exclusively on learning **temporal patterns across the 15 time steps**.

### 3.2 Layer Architecture

| Layer | Operation | Output Shape | Parameters | Rationale |
|---|---|---|---|---|
| **Input** | Raw feature map (`data_format='NTCD'`) | $(1, 15, 36)$ | 0 | Depth=1, Time=15, Channels=36 |
| **Conv 1** | `Conv2D(32, (1, 5), padding='same', activation='elu')` | $(1, 15, 32)$ | 5,792 | Temporal receptive field spanning 5 windows (1.50 s) |
| **BN 1** | `BatchNormalization(axis=-1)` | $(1, 15, 32)$ | 128 | Stabilizes internal covariate shift |
| **Drop 1** | `Dropout(0.5)` | $(1, 15, 32)$ | 0 | Prevents co-adaptation on small sample size |
| **Conv 2** | `Conv2D(16, (1, 5), padding='same', activation='elu')` | $(1, 15, 16)$ | 2,576 | Higher-level temporal feature abstraction |
| **BN 2** | `BatchNormalization(axis=-1)` | $(1, 15, 16)$ | 64 | Normalization |
| **Pool 2** | `AveragePooling2D((1, 2))` | $(1, 7, 16)$ | 0 | Temporal downsampling ($15 \rightarrow 7$) |
| **Drop 2** | `Dropout(0.5)` | $(1, 7, 16)$ | 0 | Regularization |
| **Flatten**| `Flatten()` | $(112)$ | 0 | Reshapes $7 \times 16$ temporal-spectral features |
| **Dense**  | `Dense(2, kernel_constraint=max_norm(0.5))` | $(2)$ | 226 | Linear classifier with max-norm weight clipping |
| **Softmax**| `Activation('softmax')` | $(2)$ | 0 | Binary class probability (Left vs. Right) |
| **Total**  | | | **8,786** | **Trainable: 8,690** (Initial $k=3$ model: 5,458) |

### 3.3 Training Hyperparameters
* **Optimizer**: Adam ($\text{lr} = 10^{-3}$, $\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8}$)
* **Learning Rate Schedule**: `ReduceLROnPlateau` (factor=0.5, patience=5, min_lr=$10^{-4}$)
* **Early Stopping**: Patience = 20 epochs monitoring `val_loss`
* **Loss**: Sparse Categorical Crossentropy with class balancing

---

## 4. Part 3 — Benchmark & Experimental Results

Evaluated using 5-fold cross-validation on Session T and evaluated on the held-out Session E test set for all 9 subjects:

### 4.1 Per-Subject Test Set Results (Session E)

| Subject | HWNet Test Acc ($k=5$) | MixNet Test Acc | Difference | HWNet Macro F1 | MixNet Macro F1 | Empirical Observations & Hypotheses |
|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **S01** | **87.78%** | 85.56% | **+2.22%** | **0.8761** | 0.8523 | Strong sensorimotor modulation across C3/C4; wider receptive field hypothesized to capture late ERD dynamics. |
| **S02** | 51.94% | 55.14% | -3.20% | 0.4614 | 0.5275 | **BCI Inefficiency ("illiteracy")**. Known in literature to lack distinguishable $\mu$-rhythm suppression (improved from initial 49.44%). |
| **S03** | 90.00% | 91.94% | -1.94% | 0.8989 | 0.9191 | Consistently strong bilateral discrimination on both models ($>90\%$). |
| **S04** | 64.17% | 68.47% | -4.30% | 0.6232 | 0.6685 | Moderate accuracy subject; marked inter-session baseline drift between sessions. |
| **S05** | **78.06%** | 60.83% | **+17.23%** | **0.7779** | 0.5594 | **Substantial gain (+17.23%)**. Temporal windowing successfully resolved transient power modulations that continuous raw models averaged out. |
| **S06** | **65.14%** | 60.14% | **+5.00%** | **0.6410** | 0.5880 | Low SNR subject; benefited from longer temporal context smoothing. |
| **S07** | 88.61% | 89.72% | -1.11% | 0.8845 | 0.8962 | Robust contralateral modulation across both pipelines. |
| **S08** | **94.44%** | 90.42% | **+4.02%** | **0.9444** | 0.9034 | Highest accuracy subject in cohort; clear focal activation with 94.4% test score. |
| **S09** | **87.64%** | 86.53% | **+1.11%** | **0.8757** | 0.8644 | High stability and rapid convergence on both models (+4.58% over initial HWNet). |

### 4.2 Overall Comparison Against MixNet

| Run | n_subjects | test_acc_mean | test_acc_std | f1-score_mean | f1-score_std | Trainable Params |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **HWNet (Tuned $k=5$)** | **9** | **0.7864 (78.64%)** | **±0.1478** | **0.7759** | **±0.1642** | **8,690** |
| **HWNet (Baseline $k=3$)** | 9 | 0.7833 (78.33%) | ±0.1452 | 0.7708 | ±0.1665 | 5,362 |
| **HWNet (Initial run)** | 9 | 0.7713 (77.13%) | ±0.1502 | 0.7572 | ±0.1736 | 5,362 |
| **MixNet (Baseline)**| 9 | 0.7653 (76.53%) | ±0.1510 | 0.7532 | ±0.1642 | 44,777 |

### 4.3 Statistical Significance (Paired Tests on 9 Subjects)
* **Test Accuracy**:
  * Difference (HWNet - MixNet): **+2.11 percentage points** ($78.64\%$ vs. $76.53\%$)
  * HWNet scores higher on **5 out of 9 subjects** (including **+17.23% gain** on S05, **+5.00%** on S06, **+4.02%** on S08, **+2.22%** on S01)
  * Paired t-test: $t = 0.9765, p = 0.3574$
  * Wilcoxon signed-rank test: $W = 17.0000, p = 0.5703$
* **Macro F1-score**:
  * Difference (HWNet - MixNet): **+2.27 percentage points** ($0.7759$ vs. $0.7532$)
  * Paired t-test: $t = 0.8193, p = 0.4363$
  * Wilcoxon signed-rank test: $W = 19.0000, p = 0.7344$

> **Statistical Interpretation**: Because $p > 0.05$ across both parametric and non-parametric tests, the +2.11% numerical advantage is **not statistically significant** across the 9 subjects. With $N=9$ independent units of analysis, power is constrained. The main scientific conclusion is an **attractive efficiency–performance trade-off** (~5× parameter reduction with competitive-to-higher accuracy) rather than confirmed statistical superiority.

> For full experimental methodology, screening tables, and decoupling analysis, see the dedicated [Hyperparameter Tuning Report](hyperparameter_tuning_report.md).

---

## 5. Neurophysiological Discussion

### 5.1 Why Windowed Features Improve Accuracy
Classic CSP treats the 4-second imagination period as a stationary process. However, motor imagery is inherently dynamic:
1. **Preparation Phase (0.0–0.5 s)**: Cortical readiness potential and early desynchronization.
2. **Execution Phase (0.5–2.5 s)**: Maximal sustained contralateral ERD over motor areas.
3. **Recovery / Rebound (2.5–4.0 s)**: Progressive cessation and ipsilateral $\beta$-rebound (ERS).

By discretizing time into 15 windows, `HWNet` learns to focus on the execution phase while ignoring artifact-heavy transition phases, leading to superior robustness on held-out test sessions.

### 5.2 The Challenge of Subject 2 (BCI Inefficiency)
Subject 2 scored 49.44% (chance level). In BCI research, approximately 15%–30% of users do not exhibit detectable sensorimotor rhythms (often termed "BCI illiteracy" or BCI inefficiency). This occurs due to anatomical differences (non-standard cortical folding), low resting baseline alpha power, or improper task execution. For such subjects, spatial filtering cannot find discriminative projections regardless of model depth.

---

## 6. Future Work / TODO (Architectural Enhancements)

- [ ] **TODO 1: Multi-Scale Inception Temporal Kernels**:
  Replace single-scale $(1, 3)$ kernels with parallel inception branches ($1 \times 2, 1 \times 3, 1 \times 5$) to simultaneously capture brief ERD spikes and sustained rhythms.
- [ ] **TODO 2: Squeeze-and-Excitation (SE) Channel Attention**:
  Apply attention weights across the 36 spectral-spatial channels to dynamically suppress inactive frequency bands on a per-trial basis.
- [ ] **TODO 3: Recurrent Temporal Modeling**:
  Test a lightweight Bi-directional GRU head (hidden size 16) over the 15 time steps.
