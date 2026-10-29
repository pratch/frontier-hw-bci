# MixNet Paper Summary & TL;DR

> **Paper Title:** MixNet: Joining Force of Classical and Modern Approaches toward The Comprehensive Pipeline in Motor Imagery EEG Classification  
> **Authors:** Phairot Autthasan, Rattanaphon Chaisaen, Huy Phan, Maarten De Vos, and Theerawit Wilaiprasitporn  
> **PDF Reference:** [`docs/paper/MixNet_Paper.pdf`](file:///ist-nas/users/pratchp/homework/frontier-hw-bci/docs/paper/MixNet_Paper.pdf)

---

## 1. Executive Summary (TL;DR)

Motor Imagery EEG (MI-EEG) classification suffers from **low Signal-to-Noise Ratio (SNR)** and **high inter-subject variability**.
* **Traditional methods (e.g., FBCSP + SVM)** capture frequency-specific spatial patterns well but discard temporal dynamics and non-linear representations.
* **Pure Deep Learning (e.g., EEGNet, DeepConvNet)** attempts end-to-end learning from raw EEG but often overfits on small trial counts or struggles to identify discriminative spectral-spatial filters.

**MixNet's Core Hypothesis:**  
Combining neurophysiologically-grounded **Filter Bank Common Spatial Pattern (FBCSP)** preprocessing with a **multi-task deep representation learning network (MIN2Net)** and balancing training via **Adaptive Gradient Blending (AGB)** produces a superior, highly robust MI-EEG classification pipeline.

On the benchmark **BCI Competition IV-2a (BCIC2a)** 2-class task, MixNet achieves **~85.5% mean accuracy**, significantly outperforming standard baselines.

---

## 2. The Three Architectural Pillars of MixNet

```
[Raw EEG: 20 channels, 4.0s @ 100 Hz = 400 samples]
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ PILLAR 1: Spectral-Spatial Signal Generation (FBCSP)   │
│ - 9 Butterworth bandpass sub-bands (4–40 Hz)           │
│ - Supervised CSP spatial filtering per band            │
│ - Output: Continuous tensor (1, 400, 18)               │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ PILLAR 2: Multi-Task Architecture (MIN2Net)            │
│ - Shared Temporal Encoder (1D Conv + Pooling)          │
│ - Latent Representation z (bottleneck dimension = 18)  │
│ - Branch 1: Autoencoder Decoder -> Reconstructed xr    │
│ - Branch 2: Metric Learning -> Triplet clustering on z │
│ - Branch 3: Linear Classifier -> Softmax prediction y  │
└───────────────────────┬────────────────────────────────┘
                        │
                        ▼
┌────────────────────────────────────────────────────────┐
│ PILLAR 3: Adaptive Gradient Blending (HTS)             │
│ - Multi-loss: w_mse * L_mse + w_triplet * L_triplet    │
│               + w_ce * L_ce                            │
│ - Historical Tangent Slope (HTS) dynamically balances  │
│   loss weights based on validation convergence rates   │
└────────────────────────────────────────────────────────┘
```

### Pillar 1: Spectral-Spatial Filtering
1. **Sub-Band Decomposition**: Raw 20-channel EEG trials are decomposed into 9 frequency bands:
   `[4-8]`, `[8-12]`, `[12-16]`, `[16-20]`, `[20-24]`, `[24-28]`, `[28-32]`, `[32-36]`, `[36-40]` Hz
   using a 5th-order Butterworth filter. These bands cover $\mu$ (8–12 Hz) and $\beta$ (16–24 Hz) rhythms where contralateral Event-Related Desynchronization (ERD) occurs.
2. **CSP Filtering**: For each sub-band $b$, CSP computes spatial covariance matrices across classes and solves the generalized eigenvalue problem:
   $$\Sigma_1 W = \lambda (\Sigma_1 + \Sigma_2) W$$
   Selecting $n_{\text{components}} = 2$ spatial filters per band maximizes the variance ratio between Left and Right hand trials.
3. **Continuous Signal Retention**: Unlike classical FBCSP which computes log-variance (reducing time to a single feature vector), MixNet projects the time series:
   $$S_b = W_b^T X_b$$
   Concatenating 9 bands $\times$ 2 filters yields an **18-channel continuous time series** `(1, 400, 18)`, allowing downstream neural networks to model temporal evolution.

### Pillar 2: Multi-Task Representation Learning
The network consists of three interconnected heads sharing a common encoder:
1. **Shared Temporal Encoder**:
   - Two 1D temporal convolution layers: `Conv2D(18, (1, 64))` and `Conv2D(9, (1, 32))` with ELU activations.
   - Temporal downsampling via `AveragePooling2D((1, 4))`.
   - Projects to a compact latent bottleneck $z \in \mathbb{R}^{18}$ with max-norm kernel constraint.
2. **Reconstruction Decoder ($\mathcal{L}_{\text{mse}}$)**:
   - Expands $z$ back to reconstructed trial $x_r$ using transposed convolutions (`Conv2DTranspose`).
   - Acts as a strong regularizer, ensuring $z$ preserves informative EEG dynamics.
3. **Metric Learning ($\mathcal{L}_{\text{triplet}}$)**:
   - Applies semi-hard triplet loss on latent vectors $z$ (`margin=1.0`).
   - Encourages same-class trials to cluster tightly while pushing opposite-class representations apart in Euclidean space.
4. **Supervised Classifier ($\mathcal{L}_{\text{ce}}$)**:
   - Direct linear projection `Dense(2)` with softmax on $z$.

### Pillar 3: Adaptive Gradient Blending (AGB)
Training a multi-task network with fixed loss weights often causes auxiliary tasks (reconstruction or metric learning) to overfit or dominate the classification gradient early.
* MixNet tracks the **Historical Tangent Slope (HTS)** of validation curves over recent epochs.
* During the first 7 epochs (warmup), fixed equal weights are used.
* Afterwards, AGB dynamically scales $w_{\text{mse}}$, $w_{\text{triplet}}$, and $w_{\text{ce}}$ in real time: faster-converging/overfitting tasks receive reduced weights, prioritizing tasks that are actively generalizing.

---

## 3. Experimental Protocol in the Paper

* **Dataset**: BCI Competition IV-2a (BCIC2a), 9 subjects (`A01`–`A09`), 2 classes (Left vs. Right hand).
* **Cross-Validation**: 5-fold stratified cross-validation on Session T (training/validation).
* **Evaluation Ground Truth**: Session E is strictly held-out as the test set. Best validation checkpoint from each fold is evaluated on Session E.
* **Primary Metrics**: Test Accuracy (%) and macro F1-score.

---

## 4. Key Takeaways for Designing Your Own Pipeline (`HWNet`)

The homework asks you to formulate a hypothesis and build your own pipeline:

1. **Preprocessing (`fit_transform` & `transform`)**:
   - *What MixNet did:* 9-band Butterworth bandpass + 2-component CSP per band $\rightarrow$ shape `(1, 400, 18)`.
   - *Design spaces to explore:*
     - Frequency sub-band definitions (focusing on $\mu/\beta$ bands vs. broad bands).
     - Spatial filtering: Riemannian geometry covariance, ICA, Surface Laplacian, or CSP variations.
     - Time-frequency features: Spectrograms / Continuous Wavelet Transform (CWT) vs. raw time samples.
     - Signal scaling / robust normalization per channel.
   - *Golden Rule:* Any data-driven fitting (like CSP) **must** be fitted strictly on `X_train` and applied to `X_val` / `X_test`.

2. **Model Architecture (`HWNet.py`)**:
   - MixNet uses temporal convolutions + bottleneck latent representation.
   - You can explore:
     - Spatial attention / Channel attention (SE-Net, CBAM) across EEG electrodes.
     - Conformer / Transformer self-attention blocks across temporal windows.
     - Dilated temporal convolutions or Multi-scale receptive fields (Inception-style).
     - Lightweight single-task architectures vs. multi-task setups.

3. **Benchmarking (`benchmark.py`)**:
   - Always run paired comparisons over all 9 subjects.
   - Use the built-in paired $t$-test and Wilcoxon signed-rank test to justify whether differences are statistically meaningful.
