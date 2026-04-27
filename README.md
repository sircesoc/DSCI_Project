# FoldFlow: Fast Multi-View Flow-Matching for Cloth Folding

## Project Summary

FoldFlow is a DSCI-498 project focused on robot learning for garment folding. The project studies whether a fast generative policy based on **Optimal Transport Conditional Flow Matching (OT-CFM)** can improve cloth-folding performance relative to standard diffusion-style action policies. The central idea is to combine multi-view cloth perception, transformer-based action generation, and closed-loop replanning for deformable object manipulation.

Cloth folding is a difficult robotics problem because garments are deformable, self-occluding, and highly sensitive to grasping errors. A policy that performs well on clean demonstrations can fail when the shirt becomes twisted, partially folded, or visually ambiguous. FoldFlow is designed to address this by generating action chunks efficiently and by providing a foundation for future out-of-distribution (OOD) failure recovery.

## Data

This project uses the **LeHome Simulation Challenge** as the main benchmark and data source. The challenge is built on an IsaacLab-based simulation platform and focuses on folding garments in simulation. The official garment categories are:

- long-sleeved tops  
- short-sleeved tops  
- long pants  
- shorts  

For each garment category, the challenge provides **10 training garments and 2 test garments**. The official simulated teleoperation dataset contains **1000 total samples**, with **250 samples per category**. The LeHome training guide also provides official baseline configurations for **ACT**, **Diffusion Policy**, and **SmolVLA**.

## Goals

1. Build a working FoldFlow policy within the LeRobot ecosystem  
2. Compare FoldFlow against imitation-learning, diffusion, and VLA baselines  
3. Evaluate whether OT-CFM reduces inference cost relative to diffusion policies  
4. Test whether multi-view perception improves cloth-state estimation  
5. Develop a path toward OOD failure detection and recovery for cloth folding  

## Method Overview

FoldFlow combines:

- **Multi-view RGB** from top, left, and right cameras  
- A **shared visual backbone** with cross-view fusion  
- A **DiT-style conditional transformer** for action-sequence generation  
- **OT-CFM** to generate action trajectories in fewer integration steps than standard diffusion  

The model predicts a 32-step action horizon and executes the first 16 actions before replanning, enabling closed-loop control.

## Links

- **LeHome Simulation Challenge:** https://lehome-challenge.com/simulation-challenge
- **LeHome Training Guide:** [docs/training.md](docs/training.md)
- **GitHub Repository:** https://github.com/sircesoc/DSCI_Project

---

## Required Packages

Tested on Ubuntu 22.04, Python 3.11, CUDA 12.x, NVIDIA RTX 3090.

**System / framework dependencies:**

- [Isaac Sim 4.5](https://docs.omniverse.nvidia.com/isaacsim/) — physics simulator
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/) — task and env layer
- [LeRobot](https://github.com/huggingface/lerobot) — policy + dataset framework
- [`uv`](https://github.com/astral-sh/uv) for environment management (or pip)

**Python packages** (installed automatically by `uv sync` from `pyproject.toml`):

- `torch >= 2.4` (with CUDA)
- `transformers`, `safetensors`, `huggingface-hub`
- `einops`, `timm` (DINOv2 backbone)
- `hydra-core`, `omegaconf` (config system)
- `numpy`, `opencv-python`, `pyarrow`
- `pynput` (DAgger keyboard listener)
- `matplotlib`, `tqdm`, `wandb` (logging)

**Hardware (optional, DAgger only):** two SO-101 leader arms on `/dev/ttyACM0` and `/dev/ttyACM1`.

---

## How to Run the Code

### 1. Install

```bash
git clone https://github.com/sircesoc/DSCI_Project lehome-challenge
cd lehome-challenge

pip install uv
uv sync                                  # creates .venv/ and installs deps
uv pip install -e lerobot_policy_foldflow

.venv/bin/python -c "import torch; print(torch.cuda.is_available())"
```

For full install details (Isaac Sim setup, Docker option, troubleshooting) see [docs/lehome_challenge.md](docs/lehome_challenge.md).

### 2. Download assets and dataset

```bash
hf download lehome/asset_challenge          --repo-type dataset --local-dir Assets
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

### 3. Train the headline FoldFlow v8b policy

```bash
lerobot-train --config_path=configs/train_foldflow_v8b.yaml
```

Approx. 36 hours on a single RTX 3090. Checkpoints land under `outputs/train/foldflow_v8b/checkpoints/`.

### 4. Evaluate a trained checkpoint

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
    --garment_type top_short \
    --dataset_root Datasets/example/four_types_merged \
    --num_episodes 60 \
    --enable_cameras --device cuda
```

### 5. Benchmark inference latency (no simulator required)

```bash
python -m scripts.benchmark_inference \
    --checkpoint outputs/train/foldflow_v8b/checkpoints/last/pretrained_model \
    --warmup 10 --iters 100
```

Expected on RTX 3090: ~49 ms/chunk (~20 Hz throughput).

### 6. Train residual RL head on a frozen base policy

```bash
python -m scripts.residual_rl \
    --base_policy outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
    --num_episodes 200
```

### 7. DAgger collection with SO-101 leader arms

```bash
python -m scripts.dagger_collect \
    --policy_path outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
    --dataset_root Datasets/example/four_types_merged \
    --left_arm_port /dev/ttyACM0 --right_arm_port /dev/ttyACM1 \
    --garment_type top_short --garment_name Top_Short_Seen_8 \
    --num_episodes 25
```

Controls: `S` start, `T` takeover, `R` release, `N` save, `D` discard, `Esc` abort.
