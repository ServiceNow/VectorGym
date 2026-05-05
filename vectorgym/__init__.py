"""VectorGym: A Multi-Task Benchmark for SVG Code Generation and Manipulation."""

__version__ = "1.0.0"

from .validators import BaseValidator, VLLMValidator, OpenRouterValidator
from .core.registry import register_validator, get_validator

__all__ = [
    'BaseValidator',
    'VLLMValidator',
    'OpenRouterValidator',
    'register_validator',
    'get_validator',
]

