"""Command-line interface for VectorGym evaluation."""

from omegaconf import OmegaConf
from .core.config import build_config_from_presets
from .core.registry import get_validator


def main():
    """Main CLI entry point."""
    cli_conf = OmegaConf.from_cli()
    
    # Load from config file or build from presets
    if 'config' in cli_conf:
        config = OmegaConf.load(cli_conf.pop('config'))
    elif 'task' in cli_conf or 'backend' in cli_conf:
        task = cli_conf.pop('task', None)
        backend = cli_conf.pop('backend', None)
        model_name = cli_conf.pop('model_name', None)
        
        config = build_config_from_presets(task=task, backend=backend)
        
        if model_name:
            config.model_name = model_name
    else:
        _show_usage()
        return
    
    # Apply CLI overrides
    if cli_conf:
        config = OmegaConf.merge(config, cli_conf)
    
    # Validate required fields
    if not hasattr(config, 'task') or not config.task:
        raise ValueError("Missing 'task'. Use: task=text2svg, task=sketch2svg, or task=svg_editing")
    
    if not hasattr(config, 'backend') or not config.backend:
        raise ValueError("Missing 'backend'. Use: backend=vllm or backend=openrouter")
    
    if not hasattr(config, 'model_name') or not config.model_name:
        raise ValueError("Missing 'model_name'. Specify model path/name for your backend")
    
    # Get and run validator
    validator_class = get_validator(config.backend)
    if not validator_class:
        from .core.registry import list_validators
        raise ValueError(f"Unknown backend '{config.backend}'. Available: {list_validators()}")
    
    validator = validator_class(config)
    validator.validate()
    
    print(f"\n✓ Results saved to: {validator.out_dir}")


def _show_usage():
    """Show usage information."""
    print("\nVectorGym CLI Usage")
    print("=" * 40)
    print("\n1. Using presets:")
    print("   python -m vectorgym.cli task=text2svg backend=openrouter model_name=gpt-4o")
    print("   python -m vectorgym.cli task=text2svg backend=vllm model_name=/path/to/model")
    print("\n2. Using custom config file:")
    print("   python -m vectorgym.cli config=path/to/your_config.yaml")
    print("\n3. With CLI overrides:")
    print("   python -m vectorgym.cli task=text2svg backend=openrouter model_name=gpt-4o dataset.num_samples=10")
    print("\nTasks: text2svg, sketch2svg, svg_editing")
    print("Backends: vllm, openrouter")


if __name__ == "__main__":
    main()

