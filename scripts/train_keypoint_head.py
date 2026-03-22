"""Train GarmentKeypointHead on four_types_merged_kp dataset.

Backbone (ResNet18 from a FoldFlow checkpoint) is frozen.
Only the GarmentKeypointHead decoder is trained.

Usage:
    python scripts/train_keypoint_head.py \\
        --config configs/train_keypoint.yaml \\
        [--foldflow_ckpt outputs/train/foldflow_v2/checkpoints/last/pretrained_model]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import yaml
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/train_keypoint.yaml")
    p.add_argument("--foldflow_ckpt", default=None,
                   help="Path to FoldFlow pretrained_model dir (overrides config).")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset wrapper — returns (top_rgb_crop, keypoints, visibility)
# ---------------------------------------------------------------------------

class KeypointDataset(torch.utils.data.Dataset):
    """Wraps a LeRobotDataset, returning top_rgb + keypoints for head training."""

    CROP_SIZE = 224
    IMG_MEAN = [0.485, 0.456, 0.406]
    IMG_STD = [0.229, 0.224, 0.225]

    def __init__(self, lerobot_dataset):
        self.ds = lerobot_dataset
        self.center_crop = torchvision.transforms.Compose([
            torchvision.transforms.CenterCrop(self.CROP_SIZE),
            torchvision.transforms.Normalize(self.IMG_MEAN, self.IMG_STD),
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        frame = self.ds[idx]
        # top_rgb: (3, H, W) float32 in [0, 1] (LeRobot stores as float)
        rgb = frame["observation.images.top_rgb"]
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0
        elif rgb.max() > 1.0:
            rgb = rgb / 255.0

        # Channel-first if stored as (H, W, 3)
        if rgb.shape[-1] == 3:
            rgb = rgb.permute(2, 0, 1)

        rgb = self.center_crop(rgb)  # (3, 224, 224)

        kp = frame["observation.keypoints"].float()  # (6, 2) — may have -1 invalids
        visibility = ((kp >= 0).all(-1) & (kp <= 1).all(-1)).float()  # (6,)

        return rgb, kp, visibility


# ---------------------------------------------------------------------------
# Backbone loading
# ---------------------------------------------------------------------------

def load_backbone(ckpt_path: str | Path | None) -> nn.Sequential:
    """Load ResNet18 backbone from a FoldFlow checkpoint.

    Falls back to ImageNet pretrained weights if checkpoint path is None or
    the checkpoint cannot be loaded.
    """
    # Build backbone architecture (same as MultiViewClothEncoder)
    backbone_model = torchvision.models.resnet18(weights="IMAGENET1K_V1")
    backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

    if ckpt_path is not None:
        ckpt_path = Path(ckpt_path)
        # Try to load model weights from the FoldFlow policy checkpoint
        try:
            # LeRobot saves to pretrained_model/model.safetensors or model.pt
            import safetensors.torch as sf
            st_path = ckpt_path / "model.safetensors"
            if st_path.exists():
                state = sf.load_file(str(st_path))
            else:
                pt_path = ckpt_path / "model.pt"
                state = torch.load(str(pt_path), map_location="cpu")
                if "model" in state:
                    state = state["model"]

            # Extract backbone weights (keys start with "model.vision_encoder.backbone.")
            prefix = "model.vision_encoder.backbone."
            backbone_state = {
                k[len(prefix):]: v
                for k, v in state.items()
                if k.startswith(prefix)
            }
            if backbone_state:
                backbone.load_state_dict(backbone_state, strict=True)
                print(f"[KeypointTrain] Loaded backbone from {ckpt_path}")
            else:
                print(f"[KeypointTrain] No backbone keys found in checkpoint; using ImageNet weights.")
        except Exception as e:
            print(f"[KeypointTrain] Could not load checkpoint ({e}); using ImageNet weights.")

    # Freeze backbone
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    return backbone


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds_cfg = cfg["dataset"]
    lerobot_ds = LeRobotDataset(root=ds_cfg["root"])
    dataset = KeypointDataset(lerobot_ds)
    train_cfg = cfg["training"]
    loader = DataLoader(
        dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Backbone
    ckpt_path = args.foldflow_ckpt or cfg.get("pretrained_foldflow_path")
    backbone = load_backbone(ckpt_path).to(device)

    # Keypoint head
    from lerobot_policy_foldflow.modeling_foldflow import GarmentKeypointHead
    kp_cfg = cfg["keypoint_head"]
    head = GarmentKeypointHead(
        in_channels=kp_cfg.get("in_channels", 512),
        n_keypoints=kp_cfg.get("n_keypoints", 6),
    ).to(device)

    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=train_cfg["lr"],
        weight_decay=train_cfg.get("weight_decay", 1e-5),
    )

    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    total_steps = train_cfg["steps"]
    log_freq = train_cfg.get("log_freq", 100)
    save_freq = train_cfg.get("save_freq", 1000)

    step = 0
    running_loss = 0.0
    t0 = time.time()

    print(f"[KeypointTrain] Starting training: {total_steps} steps, device={device}")
    while step < total_steps:
        for rgb, kp, vis in loader:
            if step >= total_steps:
                break

            rgb = rgb.to(device, non_blocking=True)   # (B, 3, 224, 224)
            kp = kp.to(device, non_blocking=True)     # (B, 6, 2)
            vis = vis.to(device, non_blocking=True)   # (B, 6)

            with torch.no_grad():
                features = backbone(rgb)  # (B, 512, H', W')

            head.train()
            pred_coords, _ = head(features)  # (B, 6, 2)

            loss = (
                (pred_coords - kp).pow(2).sum(-1) * vis
            ).mean()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()
            step += 1

            if step % log_freq == 0:
                avg = running_loss / log_freq
                elapsed = time.time() - t0
                print(f"  step {step:6d}/{total_steps}  loss={avg:.5f}  ({elapsed:.1f}s)")
                running_loss = 0.0

            if step % save_freq == 0 or step == total_steps:
                ckpt_file = output_dir / f"checkpoint_{step:06d}.pt"
                torch.save({"step": step, "head": head.state_dict()}, str(ckpt_file))
                # Also save as "last"
                torch.save({"step": step, "head": head.state_dict()}, str(output_dir / "checkpoint.pt"))
                print(f"  Saved checkpoint to {ckpt_file}")

    print(f"[KeypointTrain] Done. Final checkpoint at {output_dir / 'checkpoint.pt'}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
