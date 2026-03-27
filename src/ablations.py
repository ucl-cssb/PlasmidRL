"""
Ablation configurations for ICML revision experiments.

Each config is a RewardConfig with specific reward components enabled/disabled.
All other training hyperparameters remain identical across ablations.
"""

from src.rewards.bioinformatics.reward_config import RewardConfig


# Production reward weights from grpo.py:121-146 (Optuna-optimized)
_PRODUCTION_BASE = dict(
    violation_penalty_factor=0,
    punish_mode=True,
    length_reward_mode=True,
    min_length=2000,
    max_length=30000,
    ideal_min_length=3000,
    ideal_max_length=6000,
    length_reward_bonus=0.7085046275614012,
    ori_min=1,
    ori_max=1,
    ori_weight=1.0,
    promoter_min=1,
    promoter_max=5,
    promoter_weight=1.0,
    terminator_min=0,
    terminator_max=2,
    terminator_weight=0.5,
    marker_min=1,
    marker_max=2,
    marker_weight=1.0,
    cds_min=1,
    cds_max=2,
    cds_weight=1.0,
    location_aware=True,
    repeat_penalty_enabled=True,
    repeat_min_length=50,
    repeat_penalty_per_region=0.1,
)


def _config(**overrides) -> RewardConfig:
    """Create a RewardConfig from production base with overrides."""
    params = {**_PRODUCTION_BASE, **overrides}
    return RewardConfig(**params)


_ABLATION_CONFIGS = {
    # Control: identical to production
    "full_reward": _config(),

    # Disable repeat penalty only
    "no_repeat_penalty": _config(
        repeat_penalty_enabled=False,
    ),

    # Disable length-based reward (length_factor always returns 1.0)
    "no_length_prior": _config(
        length_reward_mode=False,
    ),

    # Disable cassette arrangement bonus (keep CDS detection)
    "no_cassette_bonus": _config(
        location_aware=False,
    ),

    # Only CDS scoring active — all other component weights zeroed
    "cds_only": _config(
        ori_weight=0.001,
        promoter_weight=0.001,
        terminator_weight=0.001,
        marker_weight=0.001,
        cds_weight=1.0,
        length_reward_mode=False,
        repeat_penalty_enabled=False,
        location_aware=False,
    ),

    # Only length prior active — component weights at epsilon
    "length_only": _config(
        ori_weight=0.001,
        promoter_weight=0.001,
        terminator_weight=0.001,
        marker_weight=0.001,
        cds_weight=0.001,
        length_reward_mode=True,
        repeat_penalty_enabled=False,
        location_aware=False,
    ),
}

ABLATION_NAMES: list[str] = list(_ABLATION_CONFIGS.keys())


def get_ablation_config(name: str) -> RewardConfig:
    """Get a RewardConfig for the named ablation.

    Args:
        name: One of ABLATION_NAMES

    Returns:
        RewardConfig with the appropriate reward components enabled/disabled.

    Raises:
        KeyError: If name is not a valid ablation config.
    """
    if name not in _ABLATION_CONFIGS:
        valid = ", ".join(ABLATION_NAMES)
        raise KeyError(f"Unknown ablation config '{name}'. Valid configs: {valid}")
    return _ABLATION_CONFIGS[name]
