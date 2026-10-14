# HWNet: Architecture & Evaluation (Part 2 & 3)

---

## Slide 1: Title & Handoff from Preprocessing
**Title:** HWNet: A Compact Temporal CNN for Motor Imagery  
**Subtitle:** Lightweight decoding of windowed Filter-Bank CSP features  

* **Handoff from Part 1 (`HW_prep`):**
  * Input per trial: `(36, 15)` &rarr; 36 spectral-spatial channels across 15 time steps (0.5 s windows).
  * Spectral & spatial filtering is already done: 9 sub-bands (4–40 Hz) &times; 4 CSP components.
  * **HWNet's goal:** Model the **temporal evolution (15 time steps)** of ERD/ERS without overfitting.

---

## Slide 2: HWNet Architecture
**Concept:** Compact 2-Block Temporal CNN  
**Total Parameters:** 5,458 (5,362 trainable)  

```
[Input: (1, 15, 36)] 
        │
        ▼
[Block 1: Conv2D 32 (1x3) + BatchNorm + ELU + Dropout 0.5]
        │
        ▼
[Block 2: Conv2D 16 (1x3) + BatchNorm + ELU + AvgPool (1x2) + Dropout 0.5]
        │
        ▼
[Dense Head: Flatten (112) + Dense 2 + Softmax]
```

* **Small Temporal Receptive Field:** Kernel length 3 covers ~1.0 s of motor imagery—sufficient to detect ERD onset without exploding parameters.
* **Heavy Regularization:** Dual 50% Dropout, Batch Normalization, and `max_norm(0.5)` kernel constraints tailored for ~115 training trials per fold.

---

## Slide 3: Architectural Comparison: HWNet vs. MixNet

| Dimension | MixNet (Baseline) | HWNet (Ours) | Advantage |
|---|---|---|---|
| **Input Domain** | Raw signals `(1, 400, 18)` | Power envelopes `(1, 15, 36)` | Higher SNR, lower dimension |
| **Trainable Parameters** | 44,777 | **5,362** | **~8× smaller**; avoids memorizing noise |
| **Loss Formulation** | Multi-task: MSE + Triplet + CE | Single-task: Cross-Entropy | No triplet margin / loss weight tuning |
| **Temporal Kernels** | 64 and 32 samples | 3 windows (~1.0 s) | Matched to ERD latency |
| **Training Time (9 subs)** | ~18 minutes (GPU) | **~4.5 minutes (GPU)** | 4× faster iteration |

* **Takeaway:** Preprocessing already solved the spatial-spectral challenge. A simple 5k parameter CNN generalizes better across sessions than an overparameterized autoencoder.

---

## Slide 4: Benchmark on Held-Out Test Set (Session E)
*Official 5-fold cross-validation on Session T, evaluated on held-out Session E across all 9 subjects:*

| Model | Subjects | Mean Test Accuracy | Std Dev | Mean Macro F1 | Total Parameters |
|---|:---:|:---:|:---:|:---:|:---:|
| **MixNet (Baseline)** | 9 | 76.53% | ±15.10% | 0.7532 | 44,831 |
| **HWNet (Ours)** | **9** | **77.13%** | **±15.02%** | **0.7572** | **5,458** |

* **Accuracy Improvement:** **+0.60%** (wins on 4/9 subjects)
* **Macro F1 Improvement:** **+0.40%**
* **Statistical Significance:** Paired t-test $p=0.787$, Wilcoxon $p=0.734$ (demonstrating equivalent or slightly superior distribution with 8× fewer parameters).

---

## Slide 5: Per-Subject Insights & Conclusion

* **Where HWNet Wins:**
  * **Subject 5: 76.81% vs. 60.83% (+15.98%)** — MixNet’s global temporal pooling washed out S05’s brief ERD burst. HWNet's 0.5 s windows isolated it cleanly.
  * **Subject 8: 93.47% vs. 90.42% (+3.05%)** — Top performer in the cohort.
  * **Subject 3: 92.22% vs. 91.94%** — Near-ceiling bilateral motor imagery discrimination.

* **Neurophysiological Nuance (Subject 2):**
  * **Subject 2 scored 49.44% (chance level).** Known in literature as a classic "BCI-inefficient" subject lacking detectable sensorimotor $\mu$-rhythm suppression.
  * Excluding S02, HWNet achieves **80.59%** mean test accuracy.

* **Conclusion:**
  * Combining **domain-specific FBCSP windowing** with a **compact 5k-parameter temporal CNN** outperforms a complex 45k-parameter multi-task autoencoder on small EEG sample sizes.
