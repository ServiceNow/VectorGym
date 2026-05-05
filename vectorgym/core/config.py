"""Configuration loading and merging utilities."""

from pathlib import Path
from typing import Optional
from omegaconf import OmegaConf


def load_preset_config(preset_type: str, preset_name: str, config_dir: Path) -> dict:
    """
    Load a preset config (backend presets from models.yaml).
    
    Args:
        preset_type: Type of preset ('model' for backend presets)
        preset_name: Name of the preset (e.g., 'vllm', 'openrouter')
        config_dir: Directory containing config files
    
    Returns:
        Preset configuration dict
    """
    if preset_type == "task":
        # Tasks don't need separate configs - they're just set via CLI
        return {"task": preset_name}
    
    preset_file = f"{preset_type}s.yaml"  # models.yaml
    preset_path = config_dir / preset_file
    
    if not preset_path.exists():
        raise FileNotFoundError(f"Preset file not found: {preset_path}")
    
    presets = OmegaConf.load(preset_path)
    if preset_name not in presets:
        available = list(presets.keys())
        raise ValueError(
            f"Backend '{preset_name}' not found. "
            f"Available: {available}"
        )
    
    return presets[preset_name]


def load_example_config(example_name: str, config_dir: Path) -> dict:
    """
    Load a complete example config.
    
    Args:
        example_name: Name of the example
        config_dir: Directory containing config files
    
    Returns:
        Example configuration dict
    """
    examples_path = config_dir / "examples.yaml"
    
    if not examples_path.exists():
        raise FileNotFoundError(f"Examples file not found: {examples_path}")
    
    examples = OmegaConf.load(examples_path)
    if example_name not in examples:
        available = list(examples.keys())
        raise ValueError(
            f"Example '{example_name}' not found. Available: {available}"
        )
    
    return examples[example_name]


def build_config_from_presets(
    task: Optional[str] = None,
    backend: Optional[str] = None,
    base: str = "default",
    config_dir: Optional[Path] = None
) -> OmegaConf:
    """
    Build config by merging base + backend presets.
    
    Args:
        task: Task name (e.g., 'text2svg')
        backend: Backend preset name (e.g., 'vllm', 'openrouter')
        base: Base config name (default: "default")
        config_dir: Directory containing config files. If None, uses default.
    
    Returns:
        Merged configuration
    """
    if config_dir is None:
        # Default to configs relative to this file
        config_dir = Path(__file__).parent.parent / "configs"
    
    # Load base config
    base_path = config_dir / f"{base}.yaml"
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")
    config = OmegaConf.load(base_path)
    
    # Set task if specified
    if task:
        config.task = task
    
    # Merge backend preset if specified
    if backend:
        backend_config = load_preset_config("model", backend, config_dir)
        config = OmegaConf.merge(config, backend_config)
        # Ensure task is set
        if task:
            config.task = task
    
    return config


def resolve_example_config(example_config: dict, config_dir: Path) -> OmegaConf:
    """
    Resolve an example config that references presets.
    
    Args:
        example_config: Example configuration dict
        config_dir: Directory containing config files
    
    Returns:
        Resolved configuration
    """
    # Extract preset references
    base = example_config.pop("base", "default")
    task = example_config.pop("task", None)
    model = example_config.pop("model", None)
    
    # Build base config from presets
    config = build_config_from_presets(task=task, model=model, base=base, config_dir=config_dir)
    
    # Apply example-specific overrides
    config = OmegaConf.merge(config, example_config)
    
    return config


def load_config(config_path: Optional[str] = None) -> OmegaConf:
    """
    Load configuration from file or build from presets.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration object
    """
    if config_path:
        return OmegaConf.load(config_path)
    else:
        return build_config_from_presets()

