#!/usr/bin/env python
"""Generate augmented garment variant configs with randomized textures and scales.

For each seen garment, creates N variants with:
- Random texture from the 44 available textures
- Random scale perturbation (±15%)
- All other parameters kept identical

These configs can be used to re-record episodes with varied visual appearance,
creating a more diverse training dataset for better unseen generalization.

Usage:
    python scripts/augment_garment_variants.py \
        --n_variants 5 \
        --output_dir Assets/objects/Challenge_Garment/Augmented
"""

import argparse
import json
import os
import random
import shutil
from pathlib import Path


TEXTURE_DIR = "Assets/objects/Challenge_Garment/Release/Color_Texture"
GARMENT_DIR = "Assets/objects/Challenge_Garment/Release"
GARMENT_TYPES = ["Top_Long", "Top_Short", "Pant_Long", "Pant_Short"]


def get_all_textures(repo_root: str) -> list[str]:
    """Get all available texture USD paths."""
    tex_dir = os.path.join(repo_root, TEXTURE_DIR)
    textures = []
    for f in sorted(os.listdir(tex_dir)):
        if f.endswith(".usd"):
            textures.append(f"/Assets/objects/Challenge_Garment/Release/Color_Texture/{f}")
    return textures


def get_seen_garments(repo_root: str, garment_type: str) -> list[dict]:
    """Get all seen garment configs for a garment type."""
    base = os.path.join(repo_root, GARMENT_DIR, garment_type)
    garments = []
    for d in sorted(os.listdir(base)):
        if "Seen" not in d or not os.path.isdir(os.path.join(base, d)):
            continue
        full = os.path.join(base, d)
        json_files = [f for f in os.listdir(full) if f.endswith(".json")]
        usd_files = [f for f in os.listdir(full) if f.endswith(".usd")]
        if json_files and usd_files:
            with open(os.path.join(full, json_files[0])) as f:
                cfg = json.load(f)
            garments.append({
                "name": d,
                "dir": full,
                "config": cfg,
                "json_file": json_files[0],
                "usd_file": usd_files[0],
            })
    return garments


def create_variant(
    garment: dict,
    variant_idx: int,
    all_textures: list[str],
    output_base: str,
    scale_jitter: float = 0.15,
) -> str:
    """Create a single augmented variant of a garment."""
    name = garment["name"]
    variant_name = f"{name}_Aug{variant_idx}"
    variant_dir = os.path.join(output_base, variant_name)
    os.makedirs(variant_dir, exist_ok=True)

    # Copy the USD mesh file
    src_usd = os.path.join(garment["dir"], garment["usd_file"])
    dst_usd = os.path.join(variant_dir, garment["usd_file"])
    if not os.path.exists(dst_usd):
        shutil.copy2(src_usd, dst_usd)

    # Create modified config
    cfg = json.loads(json.dumps(garment["config"]))  # deep copy

    # Randomize texture
    new_texture = random.choice(all_textures)
    cfg["visual_usd_paths"] = [new_texture]

    # Randomize scale (±scale_jitter)
    orig_scale = cfg.get("scale", [0.45, 0.45, 0.45])
    jitter = 1.0 + random.uniform(-scale_jitter, scale_jitter)
    cfg["scale"] = [s * jitter for s in orig_scale]

    # Update asset path
    cfg["asset_path"] = f"/Assets/objects/Challenge_Garment/Augmented/{variant_name}/{garment['usd_file']}"

    # Save config
    dst_json = os.path.join(variant_dir, garment["json_file"])
    with open(dst_json, "w") as f:
        json.dump(cfg, f, indent=2)

    return variant_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_variants", type=int, default=5,
                        help="Number of augmented variants per seen garment")
    parser.add_argument("--output_dir", type=str,
                        default="Assets/objects/Challenge_Garment/Augmented",
                        help="Output directory for augmented configs")
    parser.add_argument("--scale_jitter", type=float, default=0.15,
                        help="Scale jitter range (±fraction)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    repo_root = os.getcwd()

    all_textures = get_all_textures(repo_root)
    print(f"Available textures: {len(all_textures)}")

    os.makedirs(args.output_dir, exist_ok=True)
    total = 0

    for gtype in GARMENT_TYPES:
        garments = get_seen_garments(repo_root, gtype)
        print(f"\n{gtype}: {len(garments)} seen garments")

        type_output = os.path.join(args.output_dir, gtype)
        os.makedirs(type_output, exist_ok=True)

        variant_names = []
        for garment in garments:
            for i in range(args.n_variants):
                vname = create_variant(
                    garment, i, all_textures, type_output, args.scale_jitter
                )
                variant_names.append(vname)
                total += 1

        # Write garment list file
        list_file = os.path.join(type_output, f"{gtype}.txt")
        with open(list_file, "w") as f:
            # Include original seen garments
            for g in garments:
                f.write(g["name"] + "\n")
            # Include augmented variants
            for vn in variant_names:
                f.write(vn + "\n")

        print(f"  Created {len(variant_names)} augmented variants")
        print(f"  Garment list: {list_file} ({len(garments) + len(variant_names)} total)")

    print(f"\nTotal augmented variants created: {total}")
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("1. Use these garment configs to re-record episodes in sim")
    print("2. Run: python -m scripts.dataset_sim record --garment_name <variant_name>")
    print("3. Or replay existing actions with new textures using the replay script")


if __name__ == "__main__":
    main()
