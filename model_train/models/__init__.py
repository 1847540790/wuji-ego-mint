"""Model registration and construction."""

from core.registry import MODELS
from models.lingbotmap import student as _lingbotmap_student  # noqa: F401


def build_model(cfg: dict):
    """Build a registered model from its configuration mapping."""
    return MODELS.build_from_cfg(cfg)

