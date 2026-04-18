#!/usr/bin/env python
"""FoldFlow: DiT + Conditional Flow Matching policy for clothes folding.

Architecture overview:
- MultiViewClothEncoder: per-view ResNet18 + SpatialAttentionPool + CrossViewFusion
- FoldFlowDiT: Diffusion Transformer with AdaLN conditioning and alternating cross-attention
- FoldFlowModel: OT-CFM training/inference loop
- FoldFlowPolicy: LeRobot PreTrainedPolicy wrapper with observation/action queues
"""

import math
from collections import deque

import einops
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot_policy_foldflow.configuration_foldflow import FoldFlowConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


# ---------------------------------------------------------------------------
# Time embedding
# ---------------------------------------------------------------------------


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal embedding for scalar t ∈ [0, 1] followed by a 2-layer MLP."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """Args:
            t: (B,) values in [0, 1].
        Returns:
            (B, dim) time embeddings.
        """
        device = t.device
        half_dim = self.dim // 2
        freq = math.log(10000) / (half_dim - 1)
        freq = torch.exp(torch.arange(half_dim, device=device) * -freq)
        emb = t.unsqueeze(-1) * freq.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Vision modules
# ---------------------------------------------------------------------------


class SpatialAttentionPool(nn.Module):
    """Learned spatial attention pooling: (B, C, H, W) → (B, out_dim).

    Projects channels, computes attention weights via a learned query vector,
    and returns a softmax-weighted sum over spatial positions.
    """

    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_dim, kernel_size=1)
        self.query = nn.Parameter(torch.randn(out_dim))

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, C, H, W) spatial feature map.
        Returns:
            (B, out_dim) pooled feature.
        """
        x = self.proj(x)  # (B, D, H, W)
        B, D, H, W = x.shape
        x_flat = x.reshape(B, D, H * W)  # (B, D, N)
        # Dot-product attention with learned query
        attn = torch.einsum("d,bdn->bn", self.query, x_flat)  # (B, N)
        attn = F.softmax(attn, dim=-1)  # (B, N)
        out = torch.einsum("bn,bdn->bd", attn, x_flat)  # (B, D)
        return out


class CrossViewFusion(nn.Module):
    """Multi-head self-attention across camera views for a single time step.

    Input/output: (B*S, V, D)  where V = number of camera views.
    """

    def __init__(self, feat_dim: int, n_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(feat_dim)
        self.attn = nn.MultiheadAttention(feat_dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(feat_dim)
        self.ffn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * 4),
            nn.GELU(),
            nn.Linear(feat_dim * 4, feat_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B*S, V, D).
        Returns:
            (B*S, V, D) fused features.
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x


class KeypointSpatialAttention(nn.Module):
    """Extract per-keypoint features from backbone feature maps via bilinear sampling.

    Samples the 7×7 (or similar) spatial feature map at keypoint UV locations,
    then compresses through a small MLP and mean-pools across keypoints.
    """

    def __init__(self, backbone_channels: int, out_dim: int, n_keypoints: int = 6):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.proj = nn.Sequential(
            nn.Linear(backbone_channels, 128),
            nn.GELU(),
            nn.Linear(128, out_dim),
        )

    def forward(
        self, feat_map: Tensor, kp_uv: Tensor, crop_offset_x: float, crop_offset_y: float,
        crop_w: int, crop_h: int, img_w: int, img_h: int,
    ) -> Tensor:
        """Args:
            feat_map: (N, C, H', W') backbone spatial features (top-view only).
            kp_uv: (N, n_kp, 2) keypoint UV in original image [0,1]. -1 = invalid.
            crop_offset_x/y, crop_w/h, img_w/h: crop transform parameters.
        Returns:
            (N, out_dim) per-sample keypoint spatial features.
        """
        N, C, fH, fW = feat_map.shape
        n_kp = kp_uv.shape[1]

        # Map image UV [0,1] → crop-relative [0,1] → grid_sample [-1,1]
        px_x = kp_uv[..., 0] * img_w  # (N, n_kp)
        px_y = kp_uv[..., 1] * img_h
        crop_x = (px_x - crop_offset_x) / crop_w  # [0,1] in crop
        crop_y = (px_y - crop_offset_y) / crop_h
        grid_x = crop_x * 2 - 1  # [-1,1]
        grid_y = crop_y * 2 - 1

        # Validity mask: invalid if uv == -1 or outside crop
        invalid = (kp_uv[..., 0] < 0) | (kp_uv[..., 1] < 0)
        outside = (crop_x < 0) | (crop_x > 1) | (crop_y < 0) | (crop_y > 1)
        invalid = invalid | outside  # (N, n_kp)

        grid = torch.stack([grid_x, grid_y], dim=-1)  # (N, n_kp, 2)
        grid = grid.unsqueeze(2)  # (N, n_kp, 1, 2) for grid_sample

        sampled = F.grid_sample(feat_map, grid, mode="bilinear", align_corners=True, padding_mode="zeros")
        # (N, C, n_kp, 1) → (N, n_kp, C)
        sampled = sampled.squeeze(-1).permute(0, 2, 1)

        # Zero out invalid keypoints
        sampled = sampled * (~invalid).unsqueeze(-1).float()

        # MLP per keypoint, then mean pool
        kp_feat = self.proj(sampled)  # (N, n_kp, out_dim)
        # Count valid keypoints for proper averaging
        n_valid = (~invalid).float().sum(dim=1, keepdim=True).clamp(min=1)  # (N, 1)
        return kp_feat.sum(dim=1) / n_valid  # (N, out_dim)


class MultiViewClothEncoder(nn.Module):
    """Encodes multi-view RGB observations into a flat feature vector per time step.

    Pipeline: shared ResNet18 backbone → SpatialAttentionPool per view →
              CrossViewFusion across views → flatten (B, S, V*D [+ kp_spatial]).
    """

    def __init__(self, config: FoldFlowConfig):
        super().__init__()

        # Optional crop
        self.crop_shape = config.crop_shape
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        images_shape = next(iter(config.image_features.values())).shape
        self.img_h, self.img_w = images_shape[1], images_shape[2]

        self._backbone_type = "resnet"  # default

        if config.vision_backbone.startswith("paligemma"):
            from transformers import PaliGemmaForConditionalGeneration
            hf_model_id = config.pretrained_backbone_weights or "google/paligemma-3b-pt-224"
            full_model = PaliGemmaForConditionalGeneration.from_pretrained(hf_model_id)
            self.backbone = full_model.vision_tower
            del full_model  # free the language model
            # Freeze entire vision tower
            self.backbone.requires_grad_(False)
            self._backbone_type = "paligemma"
            backbone_out_channels = self.backbone.config.hidden_size  # 1152
            self.vit_proj = nn.Linear(backbone_out_channels, config.vision_feature_dim)
        elif config.vision_backbone.startswith("dinov2"):
            from transformers import Dinov2Model
            hf_model_id = config.pretrained_backbone_weights or "facebook/dinov2-base"
            dino = Dinov2Model.from_pretrained(hf_model_id)
            # Freeze embeddings + first N encoder layers
            n_freeze = config.dinov2_freeze_layers
            dino.embeddings.requires_grad_(False)
            for layer in dino.encoder.layer[:n_freeze]:
                layer.requires_grad_(False)
            self.backbone = dino
            self._backbone_type = "dinov2"
            backbone_out_channels = dino.config.hidden_size  # 768 for ViT-B, 384 for ViT-S
            self.vit_proj = nn.Linear(backbone_out_channels, config.vision_feature_dim)
        else:
            # ResNet backbone (strip FC and avgpool)
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                weights=config.pretrained_backbone_weights
            )
            self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
            self._is_dinov2 = False
            dummy_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
            dummy_shape = (1, images_shape[0], *dummy_h_w)
            feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
            backbone_out_channels = feature_map_shape[0]

        self.pool = SpatialAttentionPool(backbone_out_channels, config.vision_feature_dim)
        self.cross_view_fusion = CrossViewFusion(config.vision_feature_dim, config.num_views_fusion_heads)
        self.feature_dim = config.vision_feature_dim

        # Keypoint spatial attention (v4c)
        self.kp_spatial_attn = None
        self.kp_spatial_dim = 0
        if config.keypoint_spatial_cond:
            self.kp_spatial_attn = KeypointSpatialAttention(
                backbone_out_channels, config.kp_spatial_output_dim, config.n_keypoints,
            )
            self.kp_spatial_dim = config.kp_spatial_output_dim

    def _random_crop_with_params(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Apply random crop and return (cropped_tensor, (offset_y, offset_x))."""
        _, _, H, W = x.shape
        crop_h, crop_w = self.crop_shape
        offset_y = torch.randint(0, H - crop_h + 1, (1,)).item()
        offset_x = torch.randint(0, W - crop_w + 1, (1,)).item()
        return x[:, :, offset_y:offset_y + crop_h, offset_x:offset_x + crop_w], (offset_y, offset_x)

    def forward(self, images: Tensor, kp_uv: Tensor | None = None) -> Tensor:
        """Args:
            images: (B, S, V, C, H, W) multi-view observation sequence.
            kp_uv: (B, S, n_kp, 2) keypoint UV in [0,1], or None.
        Returns:
            (B, S, V * feature_dim [+ kp_spatial_dim]) fused vision features.
        """
        B, S, V, C, H, W = images.shape
        x = images.reshape(B * S * V, C, H, W)

        crop_offset_y, crop_offset_x = 0, 0
        if self.do_crop:
            if self.training and self.kp_spatial_attn is not None:
                x, (crop_offset_y, crop_offset_x) = self._random_crop_with_params(x)
            elif self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)
                # Deterministic center crop offsets
                crop_offset_y = (H - self.crop_shape[0]) // 2
                crop_offset_x = (W - self.crop_shape[1]) // 2

        if self._backbone_type == "paligemma":
            out = self.backbone(pixel_values=x)
            patch_tokens = out.last_hidden_state  # (B*S*V, N, 1152) — no CLS token in SigLIP
            pooled = self.vit_proj(patch_tokens.mean(dim=1))  # (B*S*V, D)
            feat_map = None
        elif self._backbone_type == "dinov2":
            out = self.backbone(pixel_values=x)
            patch_tokens = out.last_hidden_state[:, 1:, :]  # drop CLS → (B*S*V, N, D)
            pooled = self.vit_proj(patch_tokens.mean(dim=1))  # (B*S*V, D)
            feat_map = None
        else:
            feat_map = self.backbone(x)  # (B*S*V, C_out, H', W')
            pooled = self.pool(feat_map)  # (B*S*V, D)

        # Keypoint spatial attention on top-view feature maps
        kp_out = None
        if self.kp_spatial_attn is not None and kp_uv is not None:
            # Extract top-view feature maps (top_rgb is first view, stride V)
            top_indices = torch.arange(0, B * S * V, V, device=feat_map.device)
            top_feat = feat_map[top_indices]  # (B*S, C_out, H', W')
            kp_flat = kp_uv.reshape(B * S, -1, 2)  # (B*S, n_kp, 2)
            crop_h = self.crop_shape[0] if self.crop_shape else H
            crop_w = self.crop_shape[1] if self.crop_shape else W
            kp_out = self.kp_spatial_attn(
                top_feat, kp_flat,
                crop_offset_x=float(crop_offset_x), crop_offset_y=float(crop_offset_y),
                crop_w=crop_w, crop_h=crop_h, img_w=self.img_w, img_h=self.img_h,
            )  # (B*S, kp_spatial_dim)

        pooled = pooled.reshape(B * S, V, self.feature_dim)  # (B*S, V, D)
        pooled = self.cross_view_fusion(pooled)  # (B*S, V, D)
        vis = pooled.reshape(B, S, V * self.feature_dim)  # (B, S, V*D)

        if kp_out is not None:
            kp_out = kp_out.reshape(B, S, self.kp_spatial_dim)  # (B, S, kp_dim)
            vis = torch.cat([vis, kp_out], dim=-1)  # (B, S, V*D + kp_dim)

        return vis


# ---------------------------------------------------------------------------
# DiT building blocks
# ---------------------------------------------------------------------------


class AdaLN(nn.Module):
    """Adaptive Layer Norm: (1 + scale) * LN(x) + shift, conditioned on cond.

    Uses zero-init output linear layer for training stability.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2),
        )
        # Zero-init: output starts as identity transform
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """Args:
            x: (B, T, D) input sequence.
            cond: (B, cond_dim) conditioning vector.
        Returns:
            (B, T, D) modulated sequence.
        """
        scale_shift = self.mlp(cond)  # (B, 2*D)
        scale, shift = scale_shift.chunk(2, dim=-1)  # each (B, D)
        scale = scale.unsqueeze(1)  # (B, 1, D)
        shift = shift.unsqueeze(1)  # (B, 1, D)
        return (1 + scale) * self.norm(x) + shift


class DiTBlock(nn.Module):
    """Single Diffusion Transformer block with AdaLN conditioning.

    Structure:
      AdaLN → Self-Attention (residual)
      [optional] AdaLN → Cross-Attention to cond_token (residual)
      AdaLN → FFN (residual)
    """

    def __init__(
        self,
        hidden_dim: int,
        n_heads: int,
        ffn_dim: int,
        dropout: float,
        use_cross_attn: bool,
    ):
        super().__init__()
        self.use_cross_attn = use_cross_attn

        self.adaLN_sa = AdaLN(hidden_dim, hidden_dim)
        self.self_attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)

        if use_cross_attn:
            self.adaLN_ca = AdaLN(hidden_dim, hidden_dim)
            self.cross_attn = nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=dropout, batch_first=True
            )

        self.adaLN_ffn = AdaLN(hidden_dim, hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
        )

    def forward(self, x: Tensor, cond: Tensor, cond_token: Tensor) -> Tensor:
        """Args:
            x: (B, T, H) action token sequence.
            cond: (B, H) conditioning scalar embedding for AdaLN.
            cond_token: (B, 1, H) conditioning token for cross-attention.
        Returns:
            (B, T, H) updated sequence.
        """
        # Self-attention
        x_norm = self.adaLN_sa(x, cond)
        sa_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + sa_out

        # Cross-attention (every other block)
        if self.use_cross_attn:
            x_norm = self.adaLN_ca(x, cond)
            ca_out, _ = self.cross_attn(x_norm, cond_token, cond_token)
            x = x + ca_out

        # FFN
        x_norm = self.adaLN_ffn(x, cond)
        x = x + self.ffn(x_norm)

        return x


# ---------------------------------------------------------------------------
# DiT denoiser
# ---------------------------------------------------------------------------


class FoldFlowDiT(nn.Module):
    """Diffusion Transformer denoiser for FoldFlow.

    Predicts the velocity field v(x_t, t, obs) for OT-CFM.

    Conditioning: vision features + robot state + sinusoidal time embedding,
    projected to a single token used for both AdaLN scale/shift and cross-attention.
    """

    def __init__(self, config: FoldFlowConfig, obs_cond_dim: int, action_dim: int):
        super().__init__()
        H = config.dit_hidden_dim

        # Project noisy action tokens to hidden dim
        self.action_proj = nn.Linear(action_dim, H)

        # Learned temporal positional encoding
        self.pos_enc = nn.Embedding(config.horizon, H)

        # Sinusoidal time embedding
        self.time_emb = SinusoidalTimeEmbedding(H)

        # Project concatenated obs + time to conditioning embedding
        self.cond_proj = nn.Linear(obs_cond_dim + H, H)

        # DiT blocks: even-indexed without cross-attn, odd-indexed with cross-attn
        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    H,
                    config.dit_n_heads,
                    config.dit_ffn_dim,
                    config.dit_dropout,
                    use_cross_attn=(i % 2 == 1),
                )
                for i in range(config.dit_n_layers)
            ]
        )

        # Output head — zero-init for stable training start
        self.out_norm = nn.LayerNorm(H)
        self.out_head = nn.Linear(H, action_dim)
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(self, x_t: Tensor, t: Tensor, obs_cond: Tensor) -> Tensor:
        """Args:
            x_t: (B, horizon, action_dim) noisy action sequence.
            t:   (B,) time values in [0, 1].
            obs_cond: (B, obs_cond_dim) flattened vision+state features.
        Returns:
            (B, horizon, action_dim) predicted velocity field.
        """
        B, T, _ = x_t.shape

        # Project action tokens and add temporal positional encoding
        x = self.action_proj(x_t)  # (B, T, H)
        positions = torch.arange(T, device=x.device)
        x = x + self.pos_enc(positions).unsqueeze(0)  # (B, T, H)

        # Build conditioning: concat obs and time embeddings → project → single token
        t_emb = self.time_emb(t)  # (B, H)
        cond = self.cond_proj(torch.cat([obs_cond, t_emb], dim=-1))  # (B, H)
        cond_token = cond.unsqueeze(1)  # (B, 1, H) for cross-attention key/value

        # Forward through DiT blocks
        for block in self.blocks:
            x = block(x, cond, cond_token)

        # Project to action space
        x = self.out_head(self.out_norm(x))  # (B, T, action_dim)
        return x


# ---------------------------------------------------------------------------
# Garment keypoint head
# ---------------------------------------------------------------------------


class GarmentKeypointHead(nn.Module):
    """Predicts 2D pixel coordinates of N garment keypoints from ResNet18 features.

    Uses heatmap soft-argmax: predict a 2D probability map per keypoint,
    then compute expected (u,v) as the weighted average of grid coordinates.
    Output coordinates are normalized to [0,1].

    Keypoint semantics (ordered by check_points vertex index list):
        0 — left sleeve tip    (grasp target phase 0)
        1 — right sleeve tip   (grasp target phase 1)
        2 — left shoulder      (success check)
        3 — right shoulder     (success check)
        4 — left hem corner    (success check)
        5 — right hem corner   (success check)
    """

    def __init__(self, in_channels: int = 512, n_keypoints: int = 6):
        super().__init__()
        self.n_keypoints = n_keypoints
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, n_keypoints, 4, stride=2, padding=1),  # (B, N_kp, 28, 28)
        )

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        """Args:
            features: (B, C, H, W) ResNet18 spatial output.
        Returns:
            coords:   (B, n_keypoints, 2) normalized uv coordinates in [0,1]
            heatmaps: (B, n_keypoints, H_out, W_out) confidence maps
        """
        heatmaps = self.decoder(features)  # (B, N_kp, H_out, W_out)
        B, N, H, W = heatmaps.shape
        flat = heatmaps.reshape(B, N, -1)
        weights = F.softmax(flat, dim=-1).reshape(B, N, H, W)
        u_grid = torch.linspace(0, 1, W, device=features.device)
        v_grid = torch.linspace(0, 1, H, device=features.device)
        u = (weights * u_grid.view(1, 1, 1, W)).sum(dim=(-2, -1))  # (B, N)
        v = (weights * v_grid.view(1, 1, H, 1)).sum(dim=(-2, -1))  # (B, N)
        coords = torch.stack([u, v], dim=-1)  # (B, N, 2)
        return coords, heatmaps


# ---------------------------------------------------------------------------
# Flow Matching model
# ---------------------------------------------------------------------------


class FoldFlowModel(nn.Module):
    """OT-CFM model wrapping the MultiViewClothEncoder and FoldFlowDiT.

    Training: interpolates between data and noise, trains DiT to predict
              the constant OT velocity field.
    Inference: Euler integration from noise (t=1) to data (t=0).
    """

    def __init__(self, config: FoldFlowConfig):
        super().__init__()
        self.config = config

        action_dim = config.action_feature.shape[0]
        state_dim = config.robot_state_feature.shape[0]
        n_views = len(config.image_features)

        self.vision_encoder = MultiViewClothEncoder(config)

        # Flattened obs conditioning dimension fed to DiT
        obs_cond_dim = config.n_obs_steps * n_views * config.vision_feature_dim
        if config.keypoint_spatial_cond:
            obs_cond_dim += config.n_obs_steps * config.kp_spatial_output_dim
        obs_cond_dim += config.n_obs_steps * state_dim
        if config.keypoint_cond:
            obs_cond_dim += config.n_obs_steps * config.n_keypoints * 2
        if config.garment_type_cond:
            self.garment_type_emb = nn.Embedding(config.n_garment_types, config.garment_type_emb_dim)
            obs_cond_dim += config.garment_type_emb_dim
        if config.phase_cond:
            self.phase_emb = nn.Embedding(config.n_phases, config.phase_emb_dim)
            obs_cond_dim += config.phase_emb_dim

        self.dit = FoldFlowDiT(config, obs_cond_dim=obs_cond_dim, action_dim=action_dim)

    def _encode_obs(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode vision and state observations into a flat conditioning vector.

        Args:
            batch: must contain OBS_STATE (B, S, state_dim) and
                   OBS_IMAGES (B, S, V, C, H, W). Optionally contains
                   "observation.keypoints" (B, S, n_kp, 2) and
                   "observation.garment_type" (B, S, 1).
        Returns:
            (B, obs_cond_dim) flat conditioning vector.
        """
        B = batch[OBS_STATE].shape[0]
        device = batch[OBS_STATE].device
        dtype = batch[OBS_STATE].dtype

        # Pass keypoints to vision encoder for spatial attention (v4c)
        kp_uv = batch.get("observation.keypoints") if self.config.keypoint_spatial_cond else None
        vis_feat = self.vision_encoder(batch[OBS_IMAGES], kp_uv=kp_uv)  # (B, S, V*D [+ kp])
        vis_feat = vis_feat.reshape(B, -1)  # (B, S*V*D)
        state_feat = batch[OBS_STATE].reshape(B, -1)  # (B, S*state_dim)
        parts = [vis_feat, state_feat]

        if self.config.keypoint_cond:
            if "observation.keypoints" in batch:
                kp = batch["observation.keypoints"].reshape(B, -1)  # (B, S*n_kp*2)
            else:
                kp_dim = self.config.n_obs_steps * self.config.n_keypoints * 2
                kp = torch.zeros(B, kp_dim, device=device, dtype=dtype)
            parts.append(kp)

        if self.config.garment_type_cond:
            if "observation.garment_type" in batch:
                # Stored as MIN_MAX-normalized float in [0,1]; reverse to integer index.
                # Shape is (B, S) for scalar storage or (B, S, 1) for sequence storage.
                gt = batch["observation.garment_type"]
                gtype_norm = gt[:, 0, 0] if gt.dim() == 3 else gt[:, 0]  # (B,)
                gtype = torch.round(
                    gtype_norm * (self.config.n_garment_types - 1)
                ).long().clamp(0, self.config.n_garment_types - 1)
            else:
                gtype = torch.zeros(B, dtype=torch.long, device=device)
            parts.append(self.garment_type_emb(gtype))  # (B, emb_dim)

        if self.config.phase_cond:
            if "observation.phase" in batch:
                ph = batch["observation.phase"]
                phase_idx = ph[:, 0, 0] if ph.dim() == 3 else ph[:, 0]  # (B,)
                phase_idx = torch.round(
                    phase_idx * (self.config.n_phases - 1)
                ).long().clamp(0, self.config.n_phases - 1)
            else:
                phase_idx = torch.zeros(B, dtype=torch.long, device=device)
            parts.append(self.phase_emb(phase_idx))  # (B, emb_dim)

        return torch.cat(parts, dim=-1)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """OT-CFM training loss.

        Interpolates x_t = (1-t)*x0 + t*x1 and trains the DiT to predict
        the constant velocity v = x1 - x0 (pointing from data to noise,
        consistent with Euler steps x_t -= v/N during inference).

        Args:
            batch: dict with OBS_STATE, OBS_IMAGES, ACTION, optionally action_is_pad.
        Returns:
            Scalar MSE loss.
        """
        assert set(batch).issuperset({OBS_STATE, OBS_IMAGES, ACTION})
        x0 = batch[ACTION]  # (B, horizon, action_dim) — clean actions
        B = x0.shape[0]

        # Sample noise and flow time
        x1 = torch.randn_like(x0)
        t = torch.rand(B, device=x0.device, dtype=x0.dtype)

        # Add small noise floor to clean actions for regularisation
        x0 = x0 + self.config.sigma_min * torch.randn_like(x0)

        # Linear interpolation: x_t = (1-t)*x0 + t*x1
        t_bcast = t.reshape(B, 1, 1)
        x_t = (1.0 - t_bcast) * x0 + t_bcast * x1

        # Constant OT velocity target: dx_t/dt = x1 - x0
        v_target = x1 - x0

        obs_cond = self._encode_obs(batch)
        v_pred = self.dit(x_t, t, obs_cond)

        loss = F.mse_loss(v_pred, v_target, reduction="none")

        # Mask loss on padded frames at episode boundaries
        if "action_is_pad" in batch:
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        # Advantage-weighted loss (AW-BC): upweight progressive frames
        if "observation.advantage_weight" in batch:
            aw = batch["observation.advantage_weight"]
            # aw shape: (B, n_obs_steps, 1) — take last obs step
            aw = aw[:, -1, 0] if aw.dim() == 3 else aw[:, 0]  # (B,)
            loss = loss * aw.reshape(B, 1, 1)

        return loss.mean()

    @torch.no_grad()
    def correct_trajectory(self, x_failed: Tensor, t_inj: float, obs_cond: Tensor) -> Tensor:
        """SDEdit: partial noise injection → re-denoising conditioned on current obs.

        Args:
            x_failed: (B, horizon, action_dim) last action chunk before drop.
            t_inj:    noise injection level [0,1]. 0=no change, 1=full re-plan.
            obs_cond: (B, obs_cond_dim) encoded current observation.
        Returns:
            (B, horizon, action_dim) corrected chunk in policy distribution.
        """
        noise = torch.randn_like(x_failed)
        x_t = (1.0 - t_inj) * x_failed + t_inj * noise

        n_steps = max(1, int(t_inj * self.config.num_flow_steps))
        dt = t_inj / n_steps

        for step in range(n_steps):
            t = t_inj - step * dt  # t_inj → ~0
            t_tensor = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=x_t.dtype)
            v = self.dit(x_t, t_tensor, obs_cond)
            x_t = x_t - v * dt  # Euler step

        return x_t

    @torch.no_grad()
    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """Euler ODE integration from noise (t=1) to clean actions (t=0).

        Args:
            batch: dict with OBS_STATE (B, n_obs_steps, state_dim) and
                   OBS_IMAGES (B, n_obs_steps, V, C, H, W).
        Returns:
            (B, n_action_steps, action_dim) action chunk.
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        B, n_obs_steps = batch[OBS_STATE].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        action_dim = self.config.action_feature.shape[0]
        obs_cond = self._encode_obs(batch)

        # Start from pure noise at t=1
        x_t = torch.randn(B, self.config.horizon, action_dim, device=device, dtype=dtype)

        N = self.config.eval_num_flow_steps or self.config.num_flow_steps
        for step in range(N):
            t_val = 1.0 - step / N  # decreasing: 1.0 → ~1/N
            t_tensor = torch.full((B,), t_val, device=device, dtype=dtype)
            v = self.dit(x_t, t_tensor, obs_cond)
            x_t = x_t - v * (1.0 / N)  # Euler step toward t=0

        # Return only the n_action_steps relevant to the current time step
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        return x_t[:, start:end]


# ---------------------------------------------------------------------------
# LeRobot policy wrapper
# ---------------------------------------------------------------------------


class FoldFlowPolicy(PreTrainedPolicy):
    """FoldFlow policy — DiT + OT-CFM for clothes folding.

    Compatible with the LeRobot training and evaluation pipeline.
    """

    config_class = FoldFlowConfig
    name = "foldflow"

    def __init__(self, config: FoldFlowConfig, **kwargs):
        super().__init__(config)
        config.validate_features()
        self.config = config
        self._queues = None
        self.last_chunk: Tensor | None = None  # (1, horizon, action_dim) — stored for SDEdit
        self.model = FoldFlowModel(config)
        self.reset()

    def get_optim_params(self):
        return self.model.parameters()

    def reset(self):
        """Clear observation and action queues. Call on env.reset()."""
        self._queues = {
            OBS_STATE: deque(maxlen=self.config.n_obs_steps),
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues[OBS_IMAGES] = deque(maxlen=self.config.n_obs_steps)
        if self.config.keypoint_cond or self.config.keypoint_spatial_cond:
            self._queues["observation.keypoints"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.garment_type_cond:
            self._queues["observation.garment_type"] = deque(maxlen=self.config.n_obs_steps)
        if "observation.advantage_weight" in self.config.input_features:
            self._queues["observation.advantage_weight"] = deque(maxlen=self.config.n_obs_steps)
        if self.config.phase_cond:
            self._queues["observation.phase"] = deque(maxlen=self.config.n_obs_steps)

    def encode_obs_from_queues(self) -> Tensor:
        """Encode current queued observations into obs_cond for SDEdit.

        Returns:
            (1, obs_cond_dim) tensor on the model's device.
        """
        batch = {}
        for k, q in self._queues.items():
            if k != ACTION and len(q) > 0:
                batch[k] = torch.stack(list(q), dim=1)
        return self.model._encode_obs(batch)

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate a full action chunk from stacked queued observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        actions = self.model.generate_actions(batch)
        self.last_chunk = actions.clone()  # (B, horizon, action_dim) for SDEdit re-entry
        return actions

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action, maintaining observation/action queues.

        - Stacks the last n_obs_steps observations.
        - Predicts a full horizon chunk when the action queue is empty.
        - Returns one action at a time, executing n_action_steps before re-planning.
        """
        if ACTION in batch:
            batch.pop(ACTION)

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        self._queues = populate_queues(self._queues, batch)

        if len(self._queues[ACTION]) == 0:
            actions = self.predict_action_chunk(batch)
            self._queues[ACTION].extend(actions.transpose(0, 1))

        return self._queues[ACTION].popleft()

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """Compute training loss for a batch."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        loss = self.model.compute_loss(batch)
        return loss, None
