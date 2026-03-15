# LeHome Challenge 2026

Challenge on Garment Manipulation Skill Learning in Household Scenarios

- **Competition Website:** https://lehome-challenge.com/

## Quick Start

> ⚠️ **IMPORTANT**: 
> For Ubuntu version and GPU-related settings, please refer to the [IsaacSim 5.1.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html). And the simulation currently only supports CPU devices.

### 1. Installation

We offer two installation methods: UV and Docker for submission and local evaluation.

- **UV:** See [UV installation guide](installation.md).
- **Docker:** See [Docker installation guide](docker_installation.md).

### 2. Assets & Data Preparation

#### Download Simulation Assets

```bash
hf download lehome/asset_challenge --repo-type dataset --local-dir Assets
```

#### Download Example Dataset

```bash
hf download lehome/dataset_challenge_merged --repo-type dataset --local-dir Datasets/example
```

For depth or per-garment data:

```bash
hf download lehome/dataset_challenge --repo-type dataset --local-dir Datasets/example
```

#### Collect Your Own Data

See [Dataset Collection and Processing Guide](datasets.md) (SO101 Leader recommended).

### 3. Train

Train using one of the pre-configured training files:

```bash
lerobot-train --config_path=configs/train_<policy>.yaml
```

**Available config files:**
- `configs/train_act.yaml` - ACT 
- `configs/train_dp.yaml` - Diffusion Policy
- `configs/train_smolvla.yaml` - SmolVLA 
- `configs/train_foldflow.yaml` - FoldFlow

**Key configuration options:**
- **Dataset path**: Update `dataset.root` to point to your dataset
- **Input/Output features**: Specify which observations and actions to use
- **Training parameters**: Adjust `batch_size`, `steps`, `save_freq`, etc.
- **Output directory**: Modify `output_dir` to save models elsewhere

See [Training Guide](training.md) for details.

### 4. Eval

**LeRobot policy:**

```bash
python -m scripts.eval \
    --policy_type lerobot \
    --policy_path outputs/train/act_top_long/checkpoints/last/pretrained_model \
    --garment_type "top_long" \
    --dataset_root Datasets/example/top_long_merged \
    --num_episodes 2 \
    --enable_cameras \
    --device cpu
```

**Custom policy:**

```bash
python -m scripts.eval \
    --policy_type custom \
    --garment_type "top_long" \
    --num_episodes 5 \
    --enable_cameras \
    --device cpu
```

#### Common Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--policy_type` | `lerobot` or `custom` | `lerobot` |
| `--policy_path` | Path to model checkpoint | - |
| `--dataset_root` | Dataset path (for metadata) | - |
| `--garment_type` | `top_long`, `top_short`, `pant_long`, `pant_short`, `custom` | `top_long` |
| `--num_episodes` | Episodes per garment | `5` |
| `--max_steps` | Max steps per episode | `600` |
| `--enable_cameras` | Enable camera rendering | - |
| `--device` | Inference device (sim: `cpu`) | `cpu` |

See [Policy evaluation guide](policy_eval.md) for more.

#### Garment Test Configuration

Evaluation uses the `Release` set under `Assets/objects/Challenge_Garment/Release`. Use `--garment_type` for a category or edit `Release_test_list.txt` and use `--garment_type custom` for specific garments.

## Submission

Submission instructions: [competition website](https://lehome-challenge.com/).

## Acknowledgments

- [Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) – simulation
- [Isaac Lab](https://isaac-sim.github.io/IsaacLab/main/index.html) – robot learning environments
- [LeRobot](https://github.com/huggingface/lerobot) – imitation learning
- [Marble](https://marble.worldlabs.ai/) – scene generation
