---
license: mit
tags:
  - motor-control
  - rehabilitation
  - trajectory-representation
  - contrastive-learning
  - transfer-learning
  - mhealth
library_name: pytorch
pipeline_tag: feature-extraction
---

# SMT-Learner

Transformer autoencoder for spatiotemporal movement trajectories (SMT), trained with
movement performance-aware contrastive learning and post-training adaptive transfer.
It produces trial-level embeddings of pediatric upper-limb reaching movements collected
from gamified iPad assessments, and those embeddings support the
exploration-exploitation (E-E) ratio, a biomarker of motor skill learning and
rehabilitation progress.

Released with *A Deep Learning Framework for Assessing Pediatric Upper-limb Motor
Control and Gamified Mobile Rehabilitation* (IEEE TNSRE). Code:
https://github.com/RaihanSabique/SMT-Learner

## Model

| | |
|---|---|
| Input | canonicalized (x, y, t) trajectory resampled to 512 points |
| Encoder | linear stem 3 -> 64, fixed sinusoidal positional encoding, 2 post-norm transformer layers (8 heads, FFN 256, dropout 0.1) |
| Embedding | mean pooling over 512 tokens -> `Z_global` in R^64 |
| Metadata | binary task-type indicator -> 2-layer MLP -> R^32, concatenated to give a 96-d fused code |
| Heads | 2-layer projection head (R^64, contrastive only) and a single dense decoder (96 -> 512x3) |
| Parameters | 260,704 (decoder 57.1%, transformer layers 38.3%) |
| Checkpoint | 1.19 MB |

The released weights are the EXP4 adaptive-transfer checkpoint: pre-trained on MSL and
fine-tuned on MP by combined cross-task and cross-subject transfer.

## Training data

- **MSL** - untrained complex unimanual motor skill learning task; 12,957 training
  trials from 54 term and preterm-born children aged 5-8.
- **MP** - untrained simple unimanual "Orbital Reaching" task; 3,840 fine-tuning trials
  from 40 children with unilateral cerebral palsy (UCP) and typically developing controls.

Splits are at the participant level, stratified by cohort, 80:10:10
train/validation/test, so validation and test trials come from held-out children.
De-identified datasets are available on OSF; both studies were approved by the
Marquette University IRB (#4376, #4667).

## Reported performance

Baseline comparison (Suppl. TABLE III of the paper):

| Model | Data | rMSE | Ep-Err | Curve-Err | T-Corr | R-Corr | S-Corr |
|---|---|---|---|---|---|---|---|
| STTraj2Vec | MSL | 0.386 | 0.095 | 3.666 | 0.982 | 0.647 | -0.797 |
| VAE | MSL | 0.412 | 0.089 | 4.758 | 0.136 | 0.096 | -0.135 |
| Seq2Seq | MSL | 0.190 | 0.215 | 3.619 | -0.996 | -0.653 | 0.819 |
| Traj-MAE | MSL | 0.111 | 0.290 | 28.304 | -0.960 | -0.638 | 0.779 |
| ConvTraj | MSL | 0.323 | 0.147 | 30.820 | 0.742 | 0.679 | 0.660 |
| **SMT-Learner** | **MSL** | **0.086** | **0.072** | **1.577** | 0.893 | 0.644 | **0.970** |
| **SMT-Learner** | **MP** | **0.001** | **0.005** | **1.720** | 0.860 | **0.910** | **0.955** |

SMT-Learner achieves the lowest reconstruction, endpoint and curvature error and the
highest success correlation on both datasets. STTraj2Vec reaches a higher completion-time
correlation but a negative success correlation, because it encodes temporal and spatial
continuity without outcome constraints.

Adaptive transfer (EXP4) reduced the total loss from 1.05 [1.04, 1.06] to 0.12 [0.11, 0.13],
an 88.6% improvement, over 5 seeds.

## Inference cost

260,704 parameters and a 1.19 MB checkpoint run on a single CPU thread (tested on iPad):
**14.4 ms** to embed one trial and **0.042 ms** for an incremental E-E update, so the
biomarker can be recomputed between trials.

## Usage

```python
from HuggingFace.Inference import SMTLearnerInference

smt = SMTLearnerInference.from_pretrained("raihan116/SMT-Learner")

# paths: list of (x, y) touch sequences; time_diff_ms: matching timestamps
embeddings = smt.embed_trials(paths, time_diff_ms)     # (n_trials, 64)
ee_ratio = smt.exploration_exploitation(embeddings)    # (n_trials,)
```

E-E hyperparameters follow the sensitivity analysis in the paper: window W=120,
decay alpha=0.05, novelty weights beta1=0.10 and beta2=0.90.

## Intended use and limitations

Intended for research on upper-limb motor assessment and rehabilitation monitoring.
This is an assessment and monitoring model; no intervention data supports it, and it is
not a diagnostic device. Results come from two planar touchscreen reaching tasks in
children aged 5-12 in one geographic region. Behaviour on substantially different
trajectory distributions, other age bands, or non-touchscreen capture is untested and
would require retraining. Embeddings are derived from de-identified trial data only.

## Citation

```bibtex
@article{mia2026smtlearner,
  title   = {A Deep Learning Framework for Assessing Pediatric Upper-limb Motor
             Control and Gamified Mobile Rehabilitation},
  author  = {Mia, Md Raihan and Ahamed, Sheikh Iqbal and Nemanich, Samuel},
  journal = {IEEE Transactions on Neural Systems and Rehabilitation Engineering},
  year    = {2026}
}
```
