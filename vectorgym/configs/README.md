# VectorGym Configuration

## Simplified Config Structure

VectorGym uses a minimal configuration setup:

### Files

- **`default.yaml`** - Base defaults (dataset, metrics, output dir, etc.)
- **`models.yaml`** - Backend presets (vllm, openrouter)

### Usage

```bash
# Basic usage - task + backend + model_name
python -m vectorgym.cli task=text2svg backend=vllm model_name=/path/to/hf/model

# With overrides
python -m vectorgym.cli task=sketch2svg backend=openrouter model_name=gpt-4o dataset.num_samples=100

# Custom config file
python -m vectorgym.cli config=my_config.yaml
```

### Tasks

Tasks are specified via CLI (no separate config needed):
- `text2svg` - Generate SVG from text
- `sketch2svg` - Generate SVG from sketch image  
- `svg_editing` - Edit SVG based on text instruction

### Backends

- **`vllm`** - Run HuggingFace models locally with VLLM
  - `model_name`: Path to HuggingFace model (e.g., `/path/to/model` or `model-name`)
  
- **`openrouter`** - Use OpenRouter API models
  - `model_name`: Model identifier (e.g., `gpt-4o`, `claude-3.5-sonnet`)

### Metrics

Metrics are automatically configured per task by the validators. You can override in `default.yaml` or via CLI.

