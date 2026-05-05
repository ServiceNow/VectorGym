"""Data preprocessing for different tasks."""

from typing import Optional
from PIL import Image
from datasets import load_dataset
from omegaconf import OmegaConf


def prepare_task_data(config: OmegaConf):
    """
    Prepare task-specific data for text2svg/sketch2svg/svg_editing.
    
    Args:
        config: Configuration object
        
    Returns:
        Prepared dataset
    """
    # Load dataset
    ds_name = getattr(config.dataset, 'dataset_name', None)
    ds_config = getattr(config.dataset, 'config_name', None)
    ds_split = getattr(config.dataset, 'split', 'test')
    
    print(f"Loading dataset: {ds_name}, config: {ds_config}, split: {ds_split}")
    data = load_dataset(ds_name, ds_config, split=ds_split)
    print(f"Loaded {len(data)} samples from {ds_split} split")
    
    # Task-specific data preparation
    task = config.task
    
    if task == "text2svg":
        print("Preparing data for text2svg task")
        data = data.filter(
            lambda x: isinstance(x.get('original_svg'), str) and
            '<svg' in x['original_svg'].lower() and
            isinstance(x.get('text_caption'), str) and
            len(x['text_caption']) > 0
        )
        
        def prepare_text_data(examples, indices):
            captions = examples['text_caption']
            prompts = [f"Convert the input text into an SVG: {cap}" for cap in captions]
            filenames = [
                str(x) if 'svg_id' in examples else f"sample_{i}"
                for i, x in zip(indices, examples.get('svg_id', indices))
            ]
            return {
                'prompt': prompts,
                'Svg': examples['original_svg'],
                'Filename': filenames
            }
        
        data = data.map(prepare_text_data, batched=True, with_indices=True, remove_columns=data.column_names)
    
    elif task == "sketch2svg":
        print("Preparing data for sketch2svg task")
        data = data.filter(
            lambda x: isinstance(x.get('original_svg'), str) and
            '<svg' in x['original_svg'].lower() and
            isinstance(x.get('sketch_image'), Image.Image)
        )
        
        def prepare_sketch_data(examples, indices):
            images = examples['sketch_image']
            prompts = ["Convert the input sketch image into an SVG"] * len(examples['original_svg'])
            filenames = [
                str(x) if 'svg_id' in examples else f"sample_{i}"
                for i, x in zip(indices, examples.get('svg_id', indices))
            ]
            return {
                'images': images,
                'prompt': prompts,
                'Svg': examples['original_svg'],
                'Filename': filenames
            }
        
        data = data.map(prepare_sketch_data, batched=True, with_indices=True, remove_columns=data.column_names)
    
    elif task == "svg_editing":
        print("Preparing data for svg_editing task")
        data = data.filter(
            lambda x: isinstance(x.get('original_svg'), str) and
            '<svg' in x['original_svg'].lower() and
            isinstance(x.get('editing_prompt'), str) and
            len(x['editing_prompt']) > 0 and
            isinstance(x.get('editing_target_svg'), str) and
            '<svg' in x['editing_target_svg'].lower()
        )
        
        def prepare_edit_data(examples, indices):
            edit_prompts = examples['editing_prompt']
            original_svgs = examples['original_svg']
            prompts = [
                f"Edit this SVG with the following instruction: {ep}\n\nOriginal SVG:\n{orig}"
                for ep, orig in zip(edit_prompts, original_svgs)
            ]
            filenames = [
                str(x) if 'svg_id' in examples else f"sample_{i}"
                for i, x in zip(indices, examples.get('svg_id', indices))
            ]
            return {
                'prompt': prompts,
                'Svg': examples['editing_target_svg'],
                'Filename': filenames
            }
        
        data = data.map(prepare_edit_data, batched=True, with_indices=True, remove_columns=data.column_names)
    
    else:
        raise ValueError(
            f"Unknown task: {task}. Supported tasks: text2svg, sketch2svg, svg_editing"
        )
    
    # Limit number of samples if specified
    if config.dataset.num_samples != -1:
        data = data.select(range(min(config.dataset.num_samples, len(data))))
        print(f"Selected {len(data)} samples for evaluation")
    
    return data

