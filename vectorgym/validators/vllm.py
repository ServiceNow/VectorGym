"""VLLM validator for running HuggingFace models locally."""

from typing import Dict, Any, List
import torch
import traceback
import gc
from PIL import Image
from transformers import AutoTokenizer
from omegaconf import OmegaConf

from vllm import LLM, SamplingParams
from .base import BaseValidator
from ..core.registry import register_validator
from ..utils.svg import robust_svg_to_pil, use_placeholder, get_svg_original_size, clean_svg
from svgpathtools import svgstr2paths


def extract_first_svg_block(text: str) -> str:
    """Extract the first SVG block from text, handling truncated SVGs."""
    if not isinstance(text, str):
        return ""
    
    lower = text.lower()
    start = lower.find("<svg")
    if start == -1:
        return ""
    
    end = lower.find("</svg>", start)
    if end != -1:
        return text[start:end + len("</svg>")]
    
    # Handle truncated SVGs
    remaining_text = text[start:]
    partial_end = -1
    for partial in ["</sv", "</s", "</"]:
        pos = remaining_text.lower().rfind(partial)
        if pos > 0:
            partial_end = max(partial_end, pos)
    
    if partial_end > 0:
        truncated_svg = remaining_text[:partial_end].rstrip()
        if not truncated_svg.lower().endswith("</svg>"):
            truncated_svg += "</svg>"
        return text[start:start + len(truncated_svg)]
    
    # Add closing tag to remaining content
    if remaining_text.strip():
        truncated_svg = remaining_text.rstrip()
        if not truncated_svg.lower().endswith("</svg>"):
            truncated_svg += "</svg>"
        return text[start:start + len(truncated_svg)]
    
    return ""


@register_validator("vllm")
class VLLMValidator(BaseValidator):
    """Validator for running HuggingFace models with VLLM."""
    
    def __init__(self, config: OmegaConf):
        """Initialize VLLM validator."""
        super().__init__(config)
        
        print(f"Loading VLLM model from: {self.model_name}")
        
        torch_dtype = getattr(config, 'torch_dtype', 'float16')
        
        try:
            self.llm = LLM(
                model=self.model_name,
                trust_remote_code=True,
                dtype=torch_dtype,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.9,
                max_model_len=8192,
                limit_mm_per_prompt={"image": 1},
            )
            print(f"VLLM model loaded successfully with dtype {torch_dtype}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            print("Tokenizer loaded for chat template processing")
            
        except Exception as e:
            print(f"Failed to load VLLM model: {e}")
            raise RuntimeError(f"Could not load VLLM model from {self.model_name}: {e}")
    
    def get_out_dir(self, config: OmegaConf) -> str:
        """Get output directory for the model."""
        return config.model.name
    
    def generate_svg(self, batch: Dict[str, Any], generate_config: Dict[str, Any]) -> List[str]:
        """Generate SVG from input batch using VLLM."""
        vllm_inputs = []
        
        try:
            if batch.get('prompt', None) is not None:
                prompt_texts = batch['prompt'] if isinstance(batch['prompt'], list) else [batch['prompt']]
                
                if self.config.task == "text2svg":
                    use_images = False
                    images = [None] * len(prompt_texts)
                else:
                    use_images = batch.get('images', None) is not None
                    if use_images:
                        images = batch['images']
                    else:
                        images = [None] * len(prompt_texts)
                        print(f"No images available for task: {self.config.task}")
            else:
                # Default sketch2svg task
                svg_field = 'Svg' if 'Svg' in batch else 'original_svg'
                if svg_field not in batch:
                    raise ValueError(
                        f"Neither 'Svg' nor 'original_svg' found in batch. "
                        f"Available keys: {list(batch.keys())}"
                    )
                
                prompt_texts = ["Convert the input sketch image into an SVG"] * len(batch[svg_field])
                use_images = True
                
                # Convert SVG to images for sketch2svg task
                images = []
                for sample in batch[svg_field]:
                    image, _ = robust_svg_to_pil(
                        sample,
                        output_width=self.config.dataset.rasterize_size,
                        output_height=self.config.dataset.rasterize_size
                    )
                    image = image.resize(
                        (self.config.dataset.render_size, self.config.dataset.render_size),
                        Image.LANCZOS
                    )
                    images.append(image)
            
            # Build VLLM input format with chat templates
            for prompt_text, image in zip(prompt_texts, images):
                if use_images and image is not None:
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image},
                                {"type": "text", "text": prompt_text},
                            ],
                        }
                    ]
                    
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    
                    vllm_input = {
                        "prompt": formatted_prompt,
                        "multi_modal_data": {"image": image}
                    }
                else:
                    # Text-only input with chat template
                    messages = [{"role": "user", "content": prompt_text}]
                    formatted_prompt = self.tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    vllm_input = {"prompt": formatted_prompt}
                
                vllm_inputs.append(vllm_input)
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=generate_config.get('temperature', 0.0),
                top_p=generate_config.get('top_p', 0.9),
                top_k=generate_config.get('top_k', 50),
                max_tokens=generate_config.get('max_length', 8096),
                n=generate_config.get('num_generations', 1),
                frequency_penalty=generate_config.get('frequency_penalty', 0.0),
                repetition_penalty=generate_config.get('repetition_penalty', 1.0),
                presence_penalty=generate_config.get('presence_penalty', 0.0),
                min_p=generate_config.get('min_p', 0.0),
            )
            
            # Generate using VLLM
            completions = self.llm.generate(vllm_inputs, sampling_params, use_tqdm=False)
            
            outputs = []
            for i, completion in enumerate(completions):
                if completion.outputs:
                    output_text = completion.outputs[0].text
                    cleaned = extract_first_svg_block(output_text)
                    if not cleaned:
                        print(f"Warning: Could not extract SVG from VLLM output for sample {i}.")
                        cleaned = ""
                    outputs.append(cleaned)
                else:
                    print(f"Warning: No output from VLLM for sample {i}.")
                    outputs.append("")
        
        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error in VLLM generation: {e}\n{tb}")
            
            svg_field = 'Svg' if 'Svg' in batch else 'original_svg'
            outputs = [""] * len(batch[svg_field])
        
        return outputs
    
    def postprocess_svg(self, text: str, filename: str = "unknown") -> Dict[str, Any]:
        """Process SVG text with validation and cleaning."""
        if not text or not text.strip():
            return {
                'svg': use_placeholder(),
                'svg_raw': text,
                'post_processed': True,
                'no_compile': True
            }
        
        try:
            svgstr2paths(text)
            return {
                'svg': text,
                'svg_raw': text,
                'post_processed': False,
                'no_compile': False
            }
        except Exception:
            try:
                h, w = get_svg_original_size(text)
                cleaned_svg = clean_svg(text, output_width=w, output_height=h)
                svgstr2paths(cleaned_svg)
                return {
                    'svg': cleaned_svg,
                    'svg_raw': text,
                    'post_processed': True,
                    'no_compile': False
                }
            except Exception:
                return {
                    'svg': use_placeholder(),
                    'svg_raw': text,
                    'post_processed': True,
                    'no_compile': True
                }
    
    def release_memory(self):
        """Release model memory."""
        if hasattr(self, 'llm') and self.llm is not None:
            del self.llm
            self.llm = None
        
        if hasattr(self, 'tokenizer') and self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

