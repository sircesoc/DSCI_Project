"""Microbenchmark for FoldFlow inference latency.

Loads a trained checkpoint, runs generate_actions with synthetic input,
and reports per-chunk latency. No simulator required.

Usage:
  python -m scripts.benchmark_inference \\
      --checkpoint outputs/train/foldflow_v8b/checkpoints/last/pretrained_model \\
      --warmup 10 --iters 100
"""

import argparse
import time

import torch

from lerobot_policy_foldflow.modeling_foldflow import FoldFlowPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="outputs/train/foldflow_v8b/checkpoints/last/pretrained_model",
        help="Path to FoldFlow pretrained_model directory.",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Loading {args.checkpoint} on {device}...")
    policy = FoldFlowPolicy.from_pretrained(args.checkpoint).to(device)
    policy.eval()

    cfg = policy.config
    B = args.batch
    n_obs = cfg.n_obs_steps

    state_dim = cfg.input_features["observation.state"].shape[0]
    img_keys = [k for k in cfg.input_features if k.startswith("observation.images.")]
    C, H, W = cfg.input_features[img_keys[0]].shape
    crop_h, crop_w = cfg.crop_shape if cfg.crop_shape else (H, W)

    def make_batch():
        b = {
            "observation.state": torch.randn(B, n_obs, state_dim, device=device),
        }
        for k in img_keys:
            b[k] = torch.rand(B, n_obs, C, crop_h, crop_w, device=device)
        if "observation.garment_type" in cfg.input_features:
            b["observation.garment_type"] = torch.zeros(B, n_obs, 1, device=device)
        return b

    print(
        f"Config: horizon={cfg.horizon}, n_action_steps={cfg.n_action_steps}, "
        f"flow_steps={cfg.eval_num_flow_steps or cfg.num_flow_steps}, "
        f"views={len(img_keys)}, img={C}x{crop_h}x{crop_w}"
    )

    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Total params: {n_params/1e6:.2f}M")

    model = policy.model  # FoldFlowModel — call generate_actions directly

    def make_stacked_batch():
        # generate_actions expects already-stacked batch with OBS_IMAGES key.
        b = {
            "observation.state": torch.randn(B, n_obs, state_dim, device=device),
        }
        # Stack image views into OBS_IMAGES (B, n_obs, V, C, H, W)
        imgs = torch.rand(B, n_obs, len(img_keys), C, crop_h, crop_w, device=device)
        b["observation.images"] = imgs
        if "observation.garment_type" in cfg.input_features:
            b["observation.garment_type"] = torch.zeros(B, n_obs, 1, device=device)
        return b

    # Warmup
    print(f"\nWarmup ({args.warmup} iters)...")
    with torch.no_grad():
        for _ in range(args.warmup):
            batch = make_stacked_batch()
            _ = model.generate_actions(batch)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Benchmark
    print(f"Benchmark ({args.iters} iters)...")
    times_ms = []
    with torch.no_grad():
        for _ in range(args.iters):
            batch = make_stacked_batch()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            _ = model.generate_actions(batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    n = len(times_ms)
    mean = sum(times_ms) / n
    p50 = times_ms[n // 2]
    p95 = times_ms[int(n * 0.95)]
    p99 = times_ms[int(n * 0.99)] if n >= 100 else times_ms[-1]
    mn = times_ms[0]
    mx = times_ms[-1]

    print("\n=== FoldFlow inference latency (full chunk) ===")
    print(f"  iters: {n}, batch: {B}")
    print(f"  min  : {mn:6.2f} ms")
    print(f"  mean : {mean:6.2f} ms   ({1000/mean:.1f} Hz)")
    print(f"  p50  : {p50:6.2f} ms   ({1000/p50:.1f} Hz)")
    print(f"  p95  : {p95:6.2f} ms")
    print(f"  p99  : {p99:6.2f} ms")
    print(f"  max  : {mx:6.2f} ms")

    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
        print(f"\nGPU: {gpu}")


if __name__ == "__main__":
    main()
