"""OpenRouter validator for API-based models."""

import os
import time
import re
import base64
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from io import BytesIO
from PIL import Image
import requests
from transformers import AutoTokenizer
from omegaconf import OmegaConf
from svgpathtools import svgstr2paths

from .base import BaseValidator
from ..core.registry import register_validator
from ..utils.svg import use_placeholder, get_svg_original_size, clean_svg


# OpenRouter model mapping
OPEN_ROUTER_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "gpt-5": "openai/gpt-5",
    "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3.7-sonnet": "anthropic/claude-3.7-sonnet",
    "claude-sonnet-4": "anthropic/claude-sonnet-4",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "qwen-2.5-vl-7b": "qwen/qwen-2.5-vl-7b-instruct",
    "qwen-2.5-vl-32b": "qwen/qwen2.5-vl-32b-instruct",
    "qwen-2.5-vl-72b": "qwen/qwen2.5-vl-72b-instruct",
    "glm-4.1-vl": "thudm/glm-4.1v-9b-thinking",
    "glm-4.5-vl": "z-ai/glm-4.5v",
}


@register_validator("openrouter")
class OpenRouterValidator(BaseValidator):
    """Validator for OpenRouter API models."""
    
    def __init__(self, config: OmegaConf):
        """Initialize OpenRouter validator."""
        # Configure task-specific metrics BEFORE calling base class
        task = config.task
        self._configure_task_metrics(config, task)
        
        # Set temperature before base class init (needed for get_out_dir)
        self.temperature = config.generation_params.get('temperature', 0.0)
        
        # Call base class constructor
        super().__init__(config)
        
        # Setup API
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENROUTER_API_KEY environment variable not set. "
                "Get an API key from https://openrouter.ai/keys"
            )
        self.default_data_format = "png"
        self.model_version = OPEN_ROUTER_MODELS.get(
            config.model_name,
            config.model_name  # Use as-is if not in mapping
        )
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.check_credit_url = "https://openrouter.ai/api/v1/auth/key"
        self.max_retries = 5
        
        # Initialize tokenizer for token counting
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("bigcode/starcoder2-7b")
        except Exception as e:
            print(f"Warning: Could not load tokenizer for token counting: {e}")
            self.tokenizer = None
        
        # Load task-specific system prompts from config file
        self.system_prompts = self._load_generation_prompts()
        
        self.update_remaining_credit()
        
        # Force batch size to 1 for API compatibility
        if config.dataset.batch_size != 1:
            print("WARNING: Batch size is not 1. Setting batch size to 1 for OpenRouter API.")
            config.dataset.batch_size = 1
    
    def _load_generation_prompts(self) -> Dict[str, str]:
        """Load generation prompts from config file."""
        prompts_path = Path(__file__).parent.parent / "configs" / "generation_prompts.yaml"
        
        if not prompts_path.exists():
            raise FileNotFoundError(
                f"Generation prompts file not found at {prompts_path}. "
                f"Please ensure configs/generation_prompts.yaml exists."
            )
        
        prompts_config = OmegaConf.load(prompts_path)
        
        prompts = {}
        for task in ["text2svg", "sketch2svg", "svg_editing"]:
            if task not in prompts_config:
                raise ValueError(
                    f"Task '{task}' not found in generation_prompts.yaml. "
                    f"Available tasks: {list(prompts_config.keys())}"
                )
            prompts[task] = prompts_config[task].system_prompt
        
        return prompts
    
    def _configure_task_metrics(self, config: OmegaConf, task: str):
        """Configure metrics based on task type."""
        vlm_judge_config = {
            'model_name': 'anthropic/claude-3-5-sonnet',
            'task': task,
            'temperature': 0.0
        }
        config.metrics.VLMJudge_config = vlm_judge_config
        
        if task == "text2svg":
            config.metrics.CLIPScore = True
            config.metrics.VLMJudge = True
            config.metrics.CountTokenLength = True
            config.metrics.ratio_post_processed = True
            config.metrics.ratio_non_compiling = True
            config.metrics.GenerationTime = True
        elif task in ["svg_editing", "sketch2svg"]:
            config.metrics.VLMJudge = True
            config.metrics.L2 = True
            config.metrics.SSIM = True
            config.metrics.DinoScore = True
            config.metrics.LPIPS = True
            config.metrics.CountTokenLength = True
            config.metrics.GenerationTime = True
    
    def get_out_dir(self, config: OmegaConf) -> str:
        """Get output directory including task name."""
        task_name = config.task
        if self.temperature != 1.0:
            model_string = f"{self.model_name}_temp{self.temperature}"
        else:
            model_string = f"{self.model_name}"
        return os.path.join(config.run.out_dir, task_name, model_string)
    
    def update_remaining_credit(self):
        """Check remaining OpenRouter credits."""
        headers = self.get_header()
        try:
            response = requests.get(self.check_credit_url, headers=headers)
            if response.status_code == 401:
                raise ValueError("Invalid OpenRouter API key")
            response.raise_for_status()
            
            try:
                response_json = response.json()
                if "data" in response_json:
                    usage_credits = response_json["data"].get("limit_remaining", -1)
                    print(f"Remaining credits: {usage_credits}")
            except Exception as e:
                print(f"Failed to parse credit response: {e}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Failed to check OpenRouter credits: {e}")
    
    def _encode_pil_image(self, img: Image.Image, data_format: str) -> str:
        """Encode PIL image to base64 data URL."""
        im_file = BytesIO()
        img.save(im_file, format=data_format.upper())
        im_bytes = im_file.getvalue()
        im_b64 = base64.b64encode(im_bytes).decode("utf-8")
        return f"data:image/{data_format};base64,{im_b64}"
    
    def get_header(self) -> Dict[str, str]:
        """Get HTTP headers for OpenRouter API."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/vectorgym/vector-gym",
            "X-Title": "VectorGym SVG Validator"
        }
    
    def _get_content_for_task(self, prompt: str, image: Optional[Image.Image] = None) -> List[Dict]:
        """Build content for OpenRouter API based on task type."""
        content = []
        task = self.config.task
        system_prompt = self.system_prompts.get(task, self.system_prompts["text2svg"])
        
        content.append({"type": "text", "text": system_prompt})
        
        if task == "text2svg":
            if image is not None:
                print("WARNING: text2svg task received an image! Ignoring.")
            content.append({"type": "text", "text": prompt})
        elif task == "sketch2svg":
            if image is not None:
                content.append({
                    "type": "image_url",
                    "image_url": {"url": self._encode_pil_image(image, self.default_data_format)}
                })
            content.append({"type": "text", "text": prompt})
        elif task == "svg_editing":
            if image is not None:
                print("WARNING: svg_editing task received an image! Ignoring.")
            content.append({"type": "text", "text": prompt})
        
        return content
    
    def _model_generate(self, url: str, headers: Dict, payload: Dict, retry_delay: int = 1) -> Optional[str]:
        """Generate response from OpenRouter API with retry logic."""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    print(f"ERROR found in response: {data}")
                    return None
                
                choices = data.get("choices", [])
                if not choices:
                    print(f"WARNING: No choices in response: {data}")
                    return None
                
                choice = choices[0]
                message = choice.get("message", {})
                
                # Try different content fields for reasoning models
                response_text = None
                if "content" in message and message["content"]:
                    response_text = message["content"]
                elif "reasoning" in message and message["reasoning"]:
                    response_text = message["reasoning"]
                elif "text" in message:
                    response_text = message["text"]
                
                if response_text and response_text.strip():
                    return response_text
                    
            except requests.exceptions.RequestException as e:
                print(f"Request failed (attempt {attempt + 1}/{self.max_retries}): {e}")
            
            if attempt < self.max_retries - 1:
                sleep_time = retry_delay * (attempt + 1)
                time.sleep(sleep_time)
        
        return None
    
    def generate_svg(self, batch: Dict[str, Any], generate_config: Dict[str, Any]) -> List[str]:
        """Generate SVG using OpenRouter API."""
        outputs = []
        generation_metadata = []
        
        prompts = batch.get('prompt', [])
        images = batch.get('images', [None] * len(prompts))
        num_generations = generate_config.get('num_generations', 1)
        
        for i, (prompt, image) in enumerate(zip(prompts, images)):
            sample_outputs = []
            
            for gen_idx in range(num_generations):
                start_time = time.time()
                token_count = 0
                
                try:
                    content = self._get_content_for_task(prompt, image)
                    headers = self.get_header()
                    
                    payload = {
                        "model": self.model_version,
                        "messages": [{"role": "user", "content": content}],
                        "max_tokens": min(2048, generate_config.get('max_length', 1024)),
                        "temperature": max(0.01, generate_config.get('temperature', 0.0)),
                    }
                    
                    # Handle reasoning models
                    reasoning_models = [
                        "openai/gpt-5", "anthropic/claude-4.0-sonnet",
                        "anthropic/claude-sonnet-4", "anthropic/claude-3.7-sonnet",
                        "google/gemini-2.5-pro", "thudm/glm-4.1v-9b-thinking", "z-ai/glm-4.5v"
                    ]
                    
                    if any(model in self.model_version for model in reasoning_models):
                        current_max_tokens = payload.get("max_tokens", 1024)
                        payload["max_tokens"] = max(4096, current_max_tokens * 2)
                        if "glm" not in self.model_version.lower():
                            payload["extra_body"] = {
                                "reasoning": {"effort": "low", "exclude": False}
                            }
                    
                    response = self._model_generate(self.url, headers, payload)
                    
                    if gen_idx < num_generations - 1:
                        time.sleep(0.5)
                    
                    if response:
                        svg_output = self._extract_code_blocks(response)
                        sample_outputs.append(svg_output.strip())
                        
                        if self.tokenizer:
                            try:
                                token_count = len(self.tokenizer.encode(svg_output))
                            except Exception:
                                token_count = 0
                    else:
                        sample_outputs.append("")
                
                except Exception as e:
                    print(f"Error generating SVG for sample {i}, generation {gen_idx+1}: {e}")
                    sample_outputs.append("")
                
                end_time = time.time()
                generation_metadata.append({
                    'time': end_time - start_time,
                    'duration': end_time - start_time,
                    'token_count': token_count
                })
            
            if i < len(prompts) - 1:
                time.sleep(1.0)
            
            outputs.extend(sample_outputs)
        
        self.last_generation_meta = generation_metadata
        return outputs
    
    def _extract_code_blocks(self, text: str) -> str:
        """Extract SVG code from response text."""
        svg_pattern = r'<svg[^>]*>.*?</svg>'
        matches = re.findall(svg_pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            return matches[0]
        
        svg_start = text.lower().find('<svg')
        if svg_start != -1:
            return text[svg_start:]
        
        return text
    
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
        """Release any held resources."""
        import gc
        gc.collect()
        
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

