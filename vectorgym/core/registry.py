"""Validator registry system for dynamic validator registration."""

from typing import Type, Dict, Optional
from .base import BaseValidator


# Global registry for validators
_validator_registry: Dict[str, Type[BaseValidator]] = {}


def register_validator(name: Optional[str] = None):
    """
    Decorator to register a validator class.
    
    Args:
        name: Optional name to register under. If None, uses class name.
    
    Example:
        @register_validator("vllm")
        class VLLMValidator(BaseValidator):
            ...
    """
    def decorator(cls: Type[BaseValidator]):
        registry_name = name or cls.__name__
        _validator_registry[registry_name] = cls
        return cls
    return decorator


def get_validator(name: str) -> Optional[Type[BaseValidator]]:
    """
    Get a validator class from the registry.
    
    Args:
        name: Name of the validator to retrieve
    
    Returns:
        Validator class or None if not found
    """
    return _validator_registry.get(name)


def list_validators() -> list:
    """List all registered validator names."""
    return list(_validator_registry.keys())

