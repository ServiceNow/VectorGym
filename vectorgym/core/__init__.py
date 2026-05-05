"""Core abstractions for VectorGym validators."""

from .registry import register_validator, get_validator
from .config import load_config, build_config_from_presets

__all__ = ['register_validator', 'get_validator', 'load_config', 'build_config_from_presets']

