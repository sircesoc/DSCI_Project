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


class MultiViewClothEncoder(nn.Module):
    """Encodes multi-view RGB observations into a flat feature vector per time step.

    Pipeline: shared ResNet18 backbone → SpatialAttentionPool per view →
              CrossViewFusion across views → flatten (B, S, V*D).
    """

    def __init__(self, config: FoldFlowConfig):
        super().__init__()

        # Shared ResNet18 backbone (strip FC and avgpool layers)
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        # Optional crop
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Dry-run to get backbone output channels
        images_shape = next(iter(config.image_features.values())).shape
        dummy_h_w = config.crop_shape if config.crop_shape is not None else images_shape[1:]
        dummy_shape = (1, images_shape[0], *dummy_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]
        backbone_out_channels = feature_map_shape[0]

        self.pool = SpatialAttentionPool(backbone_out_channels, config.vision_feature_dim)
        self.cross_view_fusion = CrossViewFusion(config.vision_feature_dim, config.num_views_fusion_heads)
        self.feature_dim = config.vision_feature_dim

    def forward(self, images: Tensor) -> Tensor:
        """Args:
            images: (B, S, V, C, H, W) multi-view observation sequence.
        Returns:
            (B, S, V * feature_dim) fused vision features.
        """
        B, S, V, C, H, W = images.shape
        x = images.reshape(B * S * V, C, H, W)

        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)

        x = self.backbone(x)  # (B*S*V, C_out, H', W')
        x = self.pool(x)  # (B*S*V, D)
        x = x.reshape(B * S, V, self.feature_dim)  # (B*S, V, D)
        x = self.cross_view_fusion(x)  # (B*S, V, D) — fused across views
        return x.reshape(B, S, V * self.feature_dim)  # (B, S, V*D)


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
        obs_cond_dim += config.n_obs_steps * state_dim

        self.dit = FoldFlowDiT(config, obs_cond_dim=obs_cond_dim, action_dim=action_dim)

    def _encode_obs(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode vision and state observations into a flat conditioning vector.

        Args:
            batch: must contain OBS_STATE (B, S, state_dim) and
                   OBS_IMAGES (B, S, V, C, H, W).
        Returns:
            (B, n_obs * (n_views * vision_feature_dim + state_dim))
        """
        B = batch[OBS_STATE].shape[0]
        vis_feat = self.vision_encoder(batch[OBS_IMAGES])  # (B, S, V*D)
        vis_feat = vis_feat.reshape(B, -1)  # (B, S*V*D)
        state_feat = batch[OBS_STATE].reshape(B, -1)  # (B, S*state_dim)
        return torch.cat([vis_feat, state_feat], dim=-1)

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

        return loss.mean()

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

        N = self.config.num_flow_steps
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

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Generate a full action chunk from stacked queued observations."""
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        return self.model.generate_actions(batch)

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
