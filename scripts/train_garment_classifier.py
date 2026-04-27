#!/usr/bin/env python
"""Train a garment-type classifier on frozen v8b vision features.

Extracts vision features from the first frame of each episode using the
frozen v8b encoder, then trains a small classifier head (Linear-ReLU-Linear)
to predict garment type (0-3). Saves the classifier weights so they can be
loaded alongside the v8b checkpoint at eval time.

Usage:
    uv run python -m scripts.train_garment_classifier \
        --policy_path outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model \
        --dataset_root /media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged \
        --device cuda \
        --epochs 50
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.transforms import v2 as T

import pyarrow.parquet as pq

from lerobot.datasets.lerobot_dataset import LeRobotDataset

from lerobot_policy_foldflow.modeling_foldflow import GarmentTypeClassifier


def load_policy_and_encoder(policy_path: str, dataset_root: str, device: str):
    """Load the v8b policy and return the frozen FoldFlowPolicy."""
    import sys
    from importlib import import_module
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    # Register local policy configs
    try:
        import_module("lerobot_policy_foldflow")
    except ModuleNotFoundError:
        pkg_src = Path(__file__).resolve().parents[1] / "lerobot_policy_foldflow" / "src"
        if pkg_src.is_dir():
            sys.path.insert(0, str(pkg_src))
            import_module("lerobot_policy_foldflow")

    metadata = LeRobotDatasetMetadata(repo_id="lehome", root=dataset_root)
    policy_cfg = PreTrainedConfig.from_pretrained(policy_path, cli_overrides={})
    policy_cfg.pretrained_path = policy_path

    # Filter metadata to match policy input features
    if hasattr(policy_cfg, "input_features"):
        expected = set(policy_cfg.input_features.keys())
        system = {"timestamp", "frame_index", "episode_index", "index", "task_index", "next.done"}
        for feat in set(metadata.features.keys()) - expected - system:
            if feat.startswith("observation."):
                del metadata.features[feat]

    policy = make_policy(policy_cfg, ds_meta=metadata)
    policy.eval()
    policy.to(torch.device(device))
    return policy


def extract_first_frame_features(
    policy,
    dataset_root: str,
    device: str,
):
    """Extract vision features from the first frame of each episode.

    Returns:
        features: (N_episodes, vis_feat_dim) tensor
        labels: (N_episodes,) tensor of garment type indices (0-3)
    """
    # Load parquet to get episode boundaries and garment types
    parquet_path = Path(dataset_root) / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(parquet_path, columns=[
        "episode_index", "frame_index", "observation.garment_type",
    ])

    # Find first frame of each episode
    episode_first_frames = {}
    for i in range(len(table)):
        ep = table["episode_index"][i].as_py()
        fi = table["frame_index"][i].as_py()
        if ep not in episode_first_frames or fi < episode_first_frames[ep]["frame_idx"]:
            gt_raw = table["observation.garment_type"][i].as_py()
            episode_first_frames[ep] = {"frame_idx": fi, "global_idx": i, "garment_type": gt_raw}

    n_episodes = len(episode_first_frames)
    print(f"Found {n_episodes} episodes")

    # Load the full dataset to access images
    dataset = LeRobotDataset(
        repo_id="repo_foldflow_v3",
        root=dataset_root,
    )

    fp = policy  # FoldFlowPolicy
    encoder = fp.model.vision_encoder
    encoder.eval()

    # Image preprocessing: match what the policy does
    crop_h, crop_w = fp.config.crop_shape
    center_crop = T.CenterCrop((crop_h, crop_w))

    all_features = []
    all_labels = []

    image_keys = sorted(fp.config.image_features.keys())
    print(f"Image keys: {image_keys}")

    with torch.no_grad():
        for ep_idx in sorted(episode_first_frames.keys()):
            info = episode_first_frames[ep_idx]
            global_idx = info["global_idx"]
            gt_raw = info["garment_type"]

            # Garment type: stored as raw float (0.0, 1.0, 2.0, 3.0) in parquet
            label = int(round(float(gt_raw)))
            label = max(0, min(fp.config.n_garment_types - 1, label))

            # Load the sample from dataset
            sample = dataset[global_idx]

            # Stack images: (V, C, H, W)
            views = []
            for key in image_keys:
                img = sample[key]  # (C, H, W) float [0,1]
                if img.dim() == 3:
                    views.append(img)
            images = torch.stack(views, dim=0)  # (V, C, H, W)

            # Add batch and sequence dims: (1, S=2, V, C, H, W)
            # Duplicate single frame to fill n_obs_steps=2 (same as LeRobot queue init)
            images = images.unsqueeze(0).unsqueeze(0).to(device)
            images = images.expand(-1, fp.config.n_obs_steps, -1, -1, -1, -1)

            # Run through vision encoder
            vis_feat = encoder(images)  # (1, S, V*D)
            vis_feat = vis_feat.reshape(1, -1)  # (1, S*V*D)

            all_features.append(vis_feat.cpu())
            all_labels.append(label)

            if (ep_idx + 1) % 100 == 0:
                print(f"  Extracted {ep_idx + 1}/{n_episodes} episodes")

    features = torch.cat(all_features, dim=0)  # (N, feat_dim)
    labels = torch.tensor(all_labels, dtype=torch.long)  # (N,)

    print(f"Feature shape: {features.shape}")
    print(f"Label distribution: {torch.bincount(labels).tolist()}")

    return features, labels


def train_classifier(
    features: torch.Tensor,
    labels: torch.Tensor,
    n_types: int = 4,
    epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 64,
    val_split: float = 0.2,
):
    """Train the garment type classifier on extracted features."""
    n = len(features)
    perm = torch.randperm(n)
    n_val = int(n * val_split)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    train_ds = TensorDataset(features[train_idx], labels[train_idx])
    val_ds = TensorDataset(features[val_idx], labels[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)

    feat_dim = features.shape[1]
    classifier = GarmentTypeClassifier(feat_dim, n_types)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        # Train
        classifier.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        for x, y in train_loader:
            logits = classifier(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(x)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += len(x)

        # Validate
        classifier.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for x, y in val_loader:
                logits = classifier(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += len(x)

        train_acc = train_correct / train_total * 100
        val_acc = val_correct / val_total * 100 if val_total > 0 else 0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in classifier.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1:3d}/{epochs} | "
                f"Train Loss: {train_loss/train_total:.4f} | "
                f"Train Acc: {train_acc:.1f}% | "
                f"Val Acc: {val_acc:.1f}%"
            )

    print(f"\nBest validation accuracy: {best_val_acc:.1f}%")
    classifier.load_state_dict(best_state)
    return classifier


def main():
    parser = argparse.ArgumentParser(description="Train garment type classifier on v8b features")
    parser.add_argument("--policy_path", type=str,
                        default="outputs/train/foldflow_v8b/checkpoints/300000/pretrained_model")
    parser.add_argument("--dataset_root", type=str,
                        default="/media/sircesoc/WD_BLACK/lehome/dataset_challenge_merged/four_types_merged")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--save_dir", type=str, default="outputs/garment_classifier")
    args = parser.parse_args()

    print("=" * 60)
    print("Garment Type Classifier Training")
    print(f"  Policy: {args.policy_path}")
    print(f"  Dataset: {args.dataset_root}")
    print(f"  Device: {args.device}")
    print("=" * 60)

    # 1. Load policy
    print("\n[1/3] Loading v8b policy...")
    policy = load_policy_and_encoder(args.policy_path, args.dataset_root, args.device)

    # 2. Extract features from first frame of each episode
    print("\n[2/3] Extracting vision features from first frames...")
    features, labels = extract_first_frame_features(
        policy, args.dataset_root, args.device,
    )

    # Free GPU memory — we only need CPU for classifier training
    del policy
    torch.cuda.empty_cache()

    # 3. Train classifier
    print("\n[3/3] Training classifier...")
    classifier = train_classifier(
        features, labels,
        n_types=4,
        epochs=args.epochs,
        lr=args.lr,
    )

    # Save
    save_path = Path(args.save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(classifier.state_dict(), save_path / "garment_classifier.pt")
    print(f"\nClassifier saved to {save_path / 'garment_classifier.pt'}")
    print(f"Feature dim: {features.shape[1]}, Classes: 4")
    print(f"To use at eval: load these weights into GarmentTypeClassifier({features.shape[1]}, 4)")


if __name__ == "__main__":
    main()
