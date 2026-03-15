# FoldFlow Architecture

**FoldFlow** is a novel robot learning policy designed for clothes folding. It combines Conditional Flow Matching (OT-CFM) with a Diffusion Transformer (DiT) denoiser and a purpose-built multi-view cloth encoder. It is implemented as a LeRobot plugin package (`lerobot_policy_foldflow`).

---

## Motivation

Standard diffusion policies (DDPM) require 100 denoising steps at inference, which is slow for real-time robot control. FoldFlow replaces the diffusion process with **Optimal Transport Conditional Flow Matching (OT-CFM)**, which learns straighter probability paths between noise and data — reducing inference to just **10 Euler steps** while maintaining action quality.

The UNet denoiser used in standard diffusion policy struggles with long-range dependencies across a 32-step action chunk. FoldFlow replaces it with a **Diffusion Transformer (DiT)**, which uses global self-attention over all 32 action tokens simultaneously.

Clothes are deformable objects whose state is ambiguous from a single viewpoint. FoldFlow uses **three cameras** (top, left, right) with a cross-view fusion module to build a richer cloth state representation.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                    FoldFlowPolicy                       │
│                                                         │
│  Observation Queue          Action Queue                │
│  ┌─────────────┐            ┌──────────────┐            │
│  │ n_obs_steps │            │ n_action_steps│            │
│  └──────┬──────┘            └──────────────┘            │
│         │                          ↑                    │
│         ▼                          │                    │
│  ┌─────────────────────────────────────────┐            │
│  │              FoldFlowModel              │            │
│  │                                         │            │
│  │  MultiViewClothEncoder                  │            │
│  │  + robot state  →  obs_cond             │            │
│  │         ↓                               │            │
│  │   FoldFlowDiT (OT-CFM)                  │            │
│  │   noise → clean action chunk            │            │
│  └─────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

**Temporal structure:**

```
Timeline:  [t-1]  [t]  |  [t+1] ... [t+15]  | ... [t+31]
            ←obs (2)→  |  ←── execute (16) ──|  (discarded)
            ←────────── horizon (32) ─────────────────────→
```

---

## Component 1: MultiViewClothEncoder

Encodes three synchronised RGB frames into a single conditioning vector per observation step.

```
Input:  (B, S, V, C, H, W)
        B = batch, S = n_obs_steps=2, V = 3 views, C=3, H=480, W=640
Output: (B, S, V × vision_feature_dim)  →  flattened to (B, S × V × D)
```

### SpatialAttentionPool

Replaces SpatialSoftmax with a learned attention pooling over spatial positions:

```
(B, C_backbone, H', W')
    │
    ▼  Conv2d(C_backbone → D)
(B, D, H', W')
    │
    ▼  reshape → (B, D, N)   where N = H'×W'
    │
    ▼  dot-product with learned query vector q ∈ ℝᴰ
(B, N)  → softmax → attention weights
    │
    ▼  weighted sum over positions
(B, D)  ← pooled feature
```

This lets the network learn *where* to look on the garment, adapting to which spatial regions carry the most information about cloth state.

### CrossViewFusion

After pooling each view independently, a transformer fuses information across the three camera views:

```
(B×S, V=3, D)
    │
    ▼  Pre-norm Multi-Head Self-Attention (across views)
    │  + residual
    ▼  Pre-norm GELU Feed-Forward Network
    │  + residual
(B×S, V=3, D)
    │
    ▼  reshape → (B, S, V×D)
```

Cross-view attention allows the model to reason about occlusion and 3D cloth geometry — e.g., correlating a fold visible from the top camera with the resulting edge visible in a side camera.

### Backbone

Shared ResNet18 (pretrained on ImageNet) with the FC and avgpool layers removed. A random/center crop of (224×224) is applied before the backbone. Backbone output is fed into `SpatialAttentionPool`.

---

## Component 2: FoldFlowDiT

A Diffusion Transformer that predicts the OT-CFM velocity field given a noisy action sequence, a time value, and the encoded observations.

```
Input:  x_t  (B, horizon=32, action_dim=12)  — noisy actions
        t    (B,)                             — flow time ∈ [0,1]
        obs_cond (B, obs_cond_dim)            — from MultiViewClothEncoder + state

Output: v    (B, 32, 12)                     — predicted velocity field
```

### Conditioning pipeline

```
obs_cond  (B, S×V×D + S×state_dim)
             │
t ──→ SinusoidalTimeEmbedding ──→ t_emb  (B, H)
             │
             ▼  concat → Linear
           cond  (B, H=512)        ← used for AdaLN scale/shift
           cond_token  (B, 1, H)   ← used as cross-attention key/value
```

### SinusoidalTimeEmbedding

```
t ∈ [0,1]  →  sin/cos basis (B, H)  →  2-layer MLP with GELU  →  (B, H)
```

### DiT Blocks (×6)

Blocks alternate between without and with cross-attention:

```
Even blocks (i=0,2,4):                 Odd blocks (i=1,3,5):
  AdaLN → Self-Attention (residual)      AdaLN → Self-Attention (residual)
  AdaLN → FFN (residual)                 AdaLN → Cross-Attention (residual)
                                         AdaLN → FFN (residual)
```

**AdaLN (Adaptive Layer Norm):**
```
cond  →  SiLU  →  Linear(H → 2H)  →  (scale, shift)
output = (1 + scale) * LayerNorm(x) + shift
```
The output linear layer is **zero-initialised**, so each block starts as an identity function — this is critical for stable training from scratch.

**Cross-attention** uses `cond_token` (B, 1, H) as both key and value. The action tokens (B, 32, H) query the single conditioning token, letting each action position selectively attend to the global observation context.

### Action tokens

```
x_t  (B, 32, action_dim=12)
  │
  ▼  Linear(12 → 512)
  │  + Embedding(horizon=32, 512)   ← learned temporal position encoding
  ▼
(B, 32, 512)  →  DiTBlock × 6  →  LayerNorm  →  Linear(512 → 12, zero-init)
  ▼
v  (B, 32, 12)
```

---

## Component 3: OT-CFM Training & Inference

### Training (Conditional Flow Matching)

Optimal Transport CFM interpolates linearly between data and noise, yielding straight probability paths and a constant-velocity training target:

```python
x0 = clean action chunk          # (B, 32, 12) — from dataset
x1 = randn_like(x0)              # (B, 32, 12) — standard Gaussian noise
t  = rand(B)                     # (B,)        — uniform in [0, 1]

x0 += sigma_min * randn_like(x0) # small noise floor (σ_min = 1e-4)

x_t      = (1-t) * x0 + t * x1  # linear interpolation
v_target = x1 - x0               # constant OT velocity (dx_t/dt)

v_pred = DiT(x_t, t, obs_cond)
loss   = MSE(v_pred, v_target)
```

The OT velocity field `x1 - x0` is constant along each trajectory — the model only needs to learn a single vector per sample, making training efficient and stable.

### Inference (Euler Integration)

Starting from pure Gaussian noise at t=1, integrate the learned velocity field backward to t=0:

```python
x_t = randn(B, 32, 12)           # start from noise

for step in range(N=10):
    t = 1.0 - step / N            # t: 1.0 → 0.1 (decreasing)
    v = DiT(x_t, t, obs_cond)
    x_t = x_t - v / N            # Euler step toward t=0

return x_t[:, 1:17]              # n_obs_steps-1 : n_obs_steps-1+n_action_steps
```

10 Euler steps is sufficient because OT paths are close to straight lines — compare with DDPM which requires 100 steps because its noising paths are curved.

---

## Full Model Diagram

```
Observations (2 timesteps × 3 cameras):
┌──────────┐ ┌──────────┐ ┌──────────┐
│ top_rgb  │ │ left_rgb │ │right_rgb │  × 2 obs steps
└────┬─────┘ └────┬─────┘ └────┬─────┘
     └────────────┼────────────┘
                  ▼
           ResNet18 (shared)
           SpatialAttentionPool
                  ▼
           CrossViewFusion
           (self-attn across 3 views)
                  ▼
         vision_feat (B, 2, 3×512)
                  │
robot_state ──────┤  flatten
(B, 2, 12)        │
                  ▼
           obs_cond (B, 3096)
                  │
           ┌──────┴──────┐
           │             │
           ▼             ▼
     SinusoidalEmb    cond_proj
        (time t)      (B, 512)
           └──────┬──────┘
                  ▼
             FoldFlowDiT
          ┌─────────────────┐
x_t ───▶  │  DiTBlock × 6   │ ──▶ v  (B, 32, 12)
(B,32,12) │  (AdaLN + SA    │
          │   + CA alt.)    │
          └─────────────────┘
                  │
          Euler: x_t -= v/N
          (10 steps)
                  │
                  ▼
          action chunk (B, 32, 12)
          slice [1:17] → execute
```

---

## Configuration & Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `n_obs_steps` | 2 | current + 1 prior frame |
| `horizon` | 32 | total predicted action length |
| `n_action_steps` | 16 | steps executed before re-planning |
| `drop_n_last_frames` | 15 | `horizon - n_action_steps - n_obs_steps + 1` |
| `vision_backbone` | resnet18 | pretrained IMAGENET1K_V1 |
| `crop_shape` | (224, 224) | random crop train / center crop eval |
| `vision_feature_dim` | 512 | SpatialAttentionPool output dim |
| `num_views_fusion_heads` | 4 | CrossViewFusion attention heads |
| `dit_hidden_dim` | 512 | transformer width |
| `dit_n_heads` | 8 | attention heads per DiT block |
| `dit_n_layers` | 6 | number of DiT blocks |
| `dit_ffn_dim` | 2048 | FFN intermediate dimension |
| `dit_dropout` | 0.1 | dropout in DiT blocks |
| `num_flow_steps` | 10 | Euler steps at inference |
| `sigma_min` | 1e-4 | training noise floor |
| `optimizer_lr` | 1e-4 | AdamW peak LR |
| `optimizer_betas` | (0.95, 0.999) | AdamW betas |
| `scheduler_warmup_steps` | 500 | cosine schedule warmup |
| `batch_size` | 16 | |
| `total_steps` | 50K | ~8 hours on RTX 3090 |

**Model size:** 48.5M parameters
**GPU memory:** ~7.4 GB at batch=16 (full 480×640 input)
**Inference cost:** 10 forward passes through DiT (vs. 100 for DDPM)

---

## Training Observations

Loss curve (MSE of velocity field prediction):

| Step | Loss | Notes |
|------|------|-------|
| 0 | ~2.0 | random init (expected for unit-variance target) |
| 500 | 0.813 | LR warmup finishing |
| 1K | 0.246 | full LR, sharp convergence |
| 5K | 0.117 | end of first epoch |
| 10K | ~0.090 | epoch 2 |
| 20K | 0.078 | epoch 4, plateau forming |

---

## Package Structure

```
lerobot_policy_foldflow/
├── pyproject.toml
└── src/lerobot_policy_foldflow/
    ├── __init__.py                   # exports + triggers plugin registration
    ├── configuration_foldflow.py     # FoldFlowConfig (registered as "foldflow")
    ├── modeling_foldflow.py          # all model code
    └── processor_foldflow.py         # make_foldflow_pre_post_processors()
```

LeRobot auto-discovers the plugin via the `lerobot_policy_` package name prefix. Importing `lerobot_policy_foldflow` triggers `@PreTrainedConfig.register_subclass("foldflow")`, making the policy available to `lerobot-train` and `lerobot-eval`.

**Install:**
```bash
pip install -e lerobot_policy_foldflow/ --no-deps
```

**Train:**
```bash
lerobot-train --config_path configs/train_foldflow.yaml
```
