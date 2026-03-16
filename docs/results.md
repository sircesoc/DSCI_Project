# Experiment Results

## Task: `top_long` folding (BiManual)

Dataset: 250 episodes, 83K frames
Hardware: NVIDIA RTX 3090 (24GB)

---

## Baseline Policies

### Diffusion Policy (DP)
- **Checkpoint**: `outputs/train/dp_top_long/`
- **Date**: 2026-03-09
- **Params**: 271M
- **Config**: batch=16, steps=30K, lr=1e-4, warmup=500

| Step | Loss | Grad norm | LR |
|------|------|-----------|-----|
| 1K | 0.252 | 2.053 | 7.5e-5 |
| 10K | 0.038 | 0.277 | 7.9e-5 |
| 20K | 0.027 | 0.224 | 2.8e-5 |
| 30K | 0.023 | 0.207 | 9.4e-8 |

> LR decayed to ~0 by step 30K. Training time: ~1h45m.

---

## FoldFlow (DiT + OT-CFM)

Architecture: ResNet18 × 3 views → SpatialAttentionPool → CrossViewFusion → FoldFlowDiT (6 blocks, 512 hidden, 8 heads). 49M params.

### v1 — Baseline FoldFlow
- **Checkpoint**: `outputs/train/foldflow/`
- **Date**: 2026-03-14
- **Config**: batch=16, steps=50K, lr=1e-4, warmup=500, no augmentation
- **Eval**: 8 episodes, 1 garment → **37.5% success rate**

| Step | Loss | Grad norm | LR |
|------|------|-----------|-----|
| 1K | 0.246 | 1.061 | 1.0e-4 |
| 10K | 0.096 | 0.567 | 9.2e-5 |
| 20K | 0.078 | 0.497 | 6.7e-5 |
| 30K | 0.066 | 0.450 | 3.6e-5 |
| 40K | 0.058 | 0.412 | 1.0e-5 |
| 50K | 0.057 | 0.405 | 8.4e-9 |

> LR hit ~0 by step 50K. Training time: ~2h30m.

**Root causes identified for 37.5%:**
1. No explicit grasp point identification — cloth treated as black-box embedding
2. No recovery from mid-execution gripper drops (OOD cascading failures)

---

### v2 — Larger batch + augmentations
- **Checkpoint**: `outputs/train/foldflow_v2/checkpoints/095000/`
- **Date**: 2026-03-15 → 2026-03-16
- **Config**: batch=32, steps=100K (stopped at 95K), lr=1e-4, warmup=1000, augmentation ON (brightness, contrast, affine ±5°)
- **Eval**: 12 garments × 8 episodes = 96 total → **49.0% success rate (+11.5pp)**

| Step | Loss | Grad norm | LR |
|------|------|-----------|-----|
| 1K | — | — | warmup |
| 10K | 0.082 | 0.408 | 9.8e-5 |
| 20K | 0.071 | 0.379 | 9.2e-5 |
| 30K | 0.064 | 0.355 | 8.5e-5 |
| 50K | 0.050 | 0.332 | 4.4e-5 |
| 75K | 0.042 | 0.298 | 1.8e-5 |
| 90K | 0.037 | 0.266 | 2.9e-6 |
| 95K | 0.037 | 0.266 | 6.9e-7 |

> Stopped at 95K (LR ~dead). Training time: ~17.5 hours.

**Per-garment eval results (checkpoint 095K):**

| Garment | Successes | Rate |
|---------|-----------|------|
| 1 | 2/8 | 25.0% |
| 2 | 7/8 | 87.5% |
| 3 | 4/8 | 50.0% |
| 4 | 4/8 | 50.0% |
| 5 | 5/8 | 62.5% |
| 6 | 1/8 | 12.5% |
| 7 | 4/8 | 50.0% |
| 8 | 3/8 | 37.5% |
| 9 | 5/8 | 62.5% |
| 10 | 6/8 | 75.0% |
| 11 | 5/8 | 62.5% |
| 12 | 1/8 | 12.5% |
| **TOTAL** | **47/96** | **49.0%** |

High per-garment variance (12.5%–87.5%) indicates remaining failures are shape/pose-dependent — exactly what keypoint conditioning targets.

---

## Planned Next Steps

| Component | Status | Expected gain |
|---|---|---|
| FoldFlow v2 retraining (larger batch + aug) | **Done** | +11.5pp → 49.0% |
| `GarmentKeypointHead` (6-point soft-argmax) | Implemented, needs data collection | Explicit sleeve/hem localisation |
| Drop recovery pipeline (DropDetector + MPPI + SDEdit) | Implemented | Handles mid-fold drops |
| FoldFlow v3 with keypoint conditioning | Planned (after keypoint head training) | +10–15pp estimated |
