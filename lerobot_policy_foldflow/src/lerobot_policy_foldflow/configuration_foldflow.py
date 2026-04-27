#!/usr/bin/env python
from dataclasses import dataclass, field

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig
from lerobot.optim.schedulers import DiffuserSchedulerConfig


@PreTrainedConfig.register_subclass("foldflow")
@dataclass
class FoldFlowConfig(PreTrainedConfig):
    """Configuration for FoldFlowPolicy.

    Uses Conditional Flow Matching (OT-CFM) with a Diffusion Transformer (DiT) denoiser
    and a Multi-View Cloth Encoder for clothes-folding manipulation.

    Args:
        n_obs_steps: Number of observation steps (current + history).
        horizon: Total action chunk length predicted by the model.
        n_action_steps: Number of actions executed per policy invocation.
        drop_n_last_frames: Frames to drop at end of episodes to avoid heavy padding.
        vision_backbone: Torchvision ResNet variant name.
        pretrained_backbone_weights: Pretrained weights identifier (e.g. "IMAGENET1K_V1").
        use_group_norm: Replace BatchNorm with GroupNorm (incompatible with pretrained weights).
        crop_shape: (H, W) random/center crop applied before backbone.
        crop_is_random: Use random crop at train time; always center crop at eval.
        vision_feature_dim: Output dimension of SpatialAttentionPool per view.
        num_views_fusion_heads: Attention heads for cross-view fusion transformer.
        dit_hidden_dim: DiT transformer hidden dimension.
        dit_n_heads: Number of attention heads in DiT blocks.
        dit_n_layers: Number of DiT blocks.
        dit_ffn_dim: Feed-forward intermediate dimension in DiT blocks.
        dit_dropout: Dropout rate inside DiT blocks.
        num_flow_steps: Number of Euler steps for ODE integration at inference.
        sigma_min: Noise floor added to clean actions during training.
        optimizer_lr: Adam learning rate.
        optimizer_betas: Adam beta coefficients.
        optimizer_weight_decay: Adam weight decay.
        scheduler_warmup_steps: Linear warmup steps for LR scheduler.
    """

    # Temporal structure
    n_obs_steps: int = 2
    horizon: int = 32
    n_action_steps: int = 16
    drop_n_last_frames: int = 15  # horizon - n_action_steps - n_obs_steps + 1

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        }
    )

    # Keypoint conditioning (raw concatenation — v4b)
    keypoint_cond: bool = False
    n_keypoints: int = 6

    # Keypoint spatial attention (bilinear sampling from backbone feature maps — v4c)
    keypoint_spatial_cond: bool = False
    kp_spatial_output_dim: int = 64

    # Garment type conditioning
    garment_type_cond: bool = False
    n_garment_types: int = 4
    garment_type_emb_dim: int = 16

    # Garment type classifier (predict type from vision at eval instead of GT)
    garment_classifier: bool = False
    garment_classifier_weight: float = 0.5

    # Phase conditioning (hierarchical planner — v5)
    phase_cond: bool = False
    n_phases: int = 3
    phase_emb_dim: int = 16

    # Vision backbone
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "IMAGENET1K_V1"
    use_group_norm: bool = False
    crop_shape: tuple[int, int] | None = (224, 224)
    crop_is_random: bool = True
    vision_feature_dim: int = 512
    num_views_fusion_heads: int = 4
    dinov2_freeze_layers: int = 8  # number of ViT encoder layers to freeze (0=none, 12=all)

    # DiT transformer
    dit_hidden_dim: int = 512
    dit_n_heads: int = 8
    dit_n_layers: int = 6
    dit_ffn_dim: int = 2048
    dit_dropout: float = 0.1

    # Flow matching
    num_flow_steps: int = 10
    eval_num_flow_steps: int | None = None  # override at inference (more steps = finer ODE)
    sigma_min: float = 1e-4

    # Optimizer presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-5
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    def validate_features(self) -> None:
        if len(self.image_features) == 0:
            raise ValueError("FoldFlowPolicy requires at least one image feature.")
        if self.robot_state_feature is None:
            raise ValueError("FoldFlowPolicy requires 'observation.state' as an input feature.")
        if self.action_feature is None:
            raise ValueError("FoldFlowPolicy requires 'action' as an output feature.")
        if self.crop_shape is not None:
            for key, ft in self.image_features.items():
                if self.crop_shape[0] > ft.shape[1] or self.crop_shape[1] > ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` {self.crop_shape} must fit within image shape {ft.shape} "
                        f"for feature '{key}'."
                    )

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )
