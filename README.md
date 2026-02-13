# 🎨 VectorGym: A Multi-Task Benchmark for SVG Code Generation and Manipulation

<div align="center">

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/XXXX.XXXXX)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-ServiceNow/VectorGym-yellow)](https://huggingface.co/datasets/ServiceNow/VectorGym)

</div>

---

## 📖 Abstract

We introduce **VectorGym**, a comprehensive multi-task benchmark for evaluating Vision-Language Models (VLMs) on Scalable Vector Graphics (SVG) code generation and manipulation. 

### Key Features:
- 🎯 **Benchmark Four SVG Tasks**: Sketch2SVG, SVG Editing, Text2SVG, and SVG Captioning
- 📊 **Gold-Standard Annotations**: Human-authored SVG edits with comprehensive annotations
- 🤖 **VLM-as-Judge**: Automatic evaluation metric validated through human correlation studies
- 🔬 **Comprehensive Evaluation**: Analysis of leading closed-source and open-source VLMs

VectorGym addresses the critical need for systematic evaluation across diverse SVG-related capabilities in the emerging field of visual code generation, establishing a new standard for measuring progress in this domain.

## ⚙️ Setup

### 1. Environment Setup

```bash
# Create and activate conda environment
conda create -n vectorgym python=3.11.3 -y
conda activate vectorgym
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

### 2. API Key Configuration

For OpenRouter models, you need to set your API key:

```bash
export OPENROUTER_API_KEY=your_key_here
export HF_TOKEN=your_token
```

---

## 🚀 Running Evaluations

### Quick Test (3 samples)

Run a small test with 3 samples:

```bash
python -m vectorgym.cli \
    task=text2svg \
    backend=openrouter \
    model_name=gpt-4o \
    dataset.num_samples=3 \
    run.out_dir=./results
```

### Full Evaluation

Run full evaluation on all samples:

```bash
python -m vectorgym.cli \
    task=text2svg \
    backend=openrouter \
    model_name=gpt-4o \
    dataset.num_samples=-1 \
    run.out_dir=./results
```

### Using VLLM Backend

For local HuggingFace models with VLLM:

```bash
python -m vectorgym.cli \
    task=text2svg \
    backend=vllm \
    model_name=/path/to/huggingface/model \
    dataset.num_samples=50 \
    run.out_dir=./results
```

### With Custom Configuration

You can also use a custom config file:

```bash
python -m vectorgym.cli config=path/to/config.yaml
```

### CLI Overrides

Override any config parameter via CLI:

```bash
python -m vectorgym.cli \
    task=sketch2svg \
    backend=openrouter \
    model_name=claude-3.5-sonnet \
    dataset.num_samples=100 \
    generation_params.temperature=0.7 \
    run.out_dir=./results
```

---

## 📋 Available Tasks

| Task | Description | Input | Output |
|------|-------------|-------|--------|
| `text2svg` | Generate SVG from text descriptions | Text prompt | SVG code |
| `sketch2svg` | Convert sketch images to SVG | Sketch image + prompt | SVG code |
| `svg_editing` | Edit SVGs with natural language | SVG + instruction | Modified SVG |

## 🔧 Backends

VectorGym supports two backends:

- **`vllm`**: Run HuggingFace models locally using VLLM
  - Requires: `model_name` should be a path to a HuggingFace model (local path or model ID)
  - Example: `backend=vllm model_name=/path/to/model` or `backend=vllm model_name=model-name`

- **`openrouter`**: Use OpenRouter API for cloud-based models
  - Requires: `OPENROUTER_API_KEY` environment variable
  - `model_name` should be a model identifier (e.g., `gpt-4o`, `claude-3.5-sonnet`)
  - Example: `backend=openrouter model_name=gpt-4o`

## 📁 Output Structure

Results are saved in the following directory structure:

```
{run.out_dir}/
└── {task}/                                 # e.g., text2svg, sketch2svg, svg_editing
    └── {model_name}_temp{temperature}/      # e.g., gpt-4o_temp0.0
        └── results_dataset_{dataset}_{timestamp}/
            ├── config.yaml                  # Run configuration
            │
            ├── {sample_id}/                 # Per-sample results
            │   ├── {sample_id}.svg          # Generated SVG
            │   ├── {sample_id}_gt.svg       # Ground truth SVG
            │   ├── {sample_id}_generated.png
            │   ├── {sample_id}_original.png
            │   ├── {sample_id}_input_prompt.txt
            │   └── metadata.json            # Sample metadata
            │
            └── results/                     # Aggregated metrics
                ├── results_avg.json         # Average metrics
                ├── all_results.csv          # Per-sample metrics
                └── comparison_grid_100.png  # Visual comparison grid
```

Default output directory is `./results` (can be changed via `run.out_dir`).

---

## 📁 Project Structure

```
vector-gym/
├── vectorgym/
│   ├── __init__.py
│   ├── cli.py                         # Main CLI entry point
│   │
│   ├── core/                          # Core abstractions
│   │   ├── base.py                    # Base validator class
│   │   ├── registry.py                # Validator registry
│   │   └── config.py                  # Config loading utilities
│   │
│   ├── validators/                     # Validator implementations
│   │   ├── base.py                    # Base validator with common functionality
│   │   ├── openrouter.py              # OpenRouter API validator
│   │   └── vllm.py                    # VLLM validator (HuggingFace models)
│   │
│   ├── metrics/                       # Evaluation metrics
│   │   ├── calculator.py              # Main metrics orchestrator
│   │   ├── compute_clip_score.py      # CLIP Score computation
│   │   ├── compute_vlm_judge.py       # VLM-as-Judge evaluation
│   │   ├── compute_l2.py              # L2/MSE computation
│   │   ├── compute_SSIM.py           # SSIM computation
│   │   ├── compute_LPIPS.py          # LPIPS computation
│   │   ├── compute_dino_score.py      # DINO Score computation
│   │   ├── count_token_length.py      # Token counting
│   │   └── compute_generation_time.py # Timing metrics
│   │
│   ├── data/                          # Data handling
│   │   ├── loader.py                  # Unified data loader
│   │   └── processor.py               # Task-specific data preprocessing
│   │
│   ├── utils/                         # Utility functions
│   │   └── svg.py                     # SVG processing utilities
│   │
│   └── configs/                       # Configuration files
│       ├── default.yaml               # Base defaults
│       ├── models.yaml                # Backend presets (vllm, openrouter)
│       └── generation_prompts.yaml    # System prompts for generation
│
├── pyproject.toml                     # Package configuration
└── README.md                          # This file
```

---

## 📄 Citation

If you use VectorGym in your research, please cite our work:

Add Citation

---

## 📜 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

We welcome contributions! Please feel free to:

1. 🐛 Report bugs or issues
2. 💡 Suggest new features or improvements
3. 🔧 Submit pull requests

For major changes, please open an issue first to discuss what you would like to change.

---

## 📧 Contact

For questions or feedback, please contact:
- 

---

<div align="center">

⭐ Star us on GitHub if you find VectorGym useful!

</div>
