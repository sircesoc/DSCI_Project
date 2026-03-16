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

## LeHome Challenge (Setup & Evaluation)

This repo is built on the **LeHome Challenge 2026** framework. For installation, asset download, training, and evaluation instructions, see:

- **[LeHome Challenge guide →](docs/lehome_challenge.md)** — installation (UV / Docker), assets, data, train/eval commands, submission  

Quick reference:

```bash
# Install (see docs/lehome_challenge.md for full instructions)
# Download assets & dataset
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example

# Train FoldFlow
lerobot-train --config_path=configs/train_foldflow.yaml

# Eval (example)
python -m scripts.eval --policy_type lerobot --policy_path <path> --garment_type "top_long" \
  --dataset_root Datasets/example/top_long_merged --num_episodes 2 --enable_cameras --device cpu
```
