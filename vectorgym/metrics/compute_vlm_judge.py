import os
import json
import requests
import time
import re
from io import BytesIO
import base64
from PIL import Image
from .base_metric import BaseMetric
from tqdm import tqdm
import math


class VLMJudgeCalculator(BaseMetric):
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.config = config or {}
        
        # OpenRouter API configuration
        self.api_key = os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            print("WARNING: OPENROUTER_API_KEY not found, VLM Judge will be disabled")
            self.api_key = None
            return  # Exit early to disable VLM Judge
        
        # Default model - use GPT-4o for VLM evaluation
        self.model_name = self.config.get('model_name', 'openai/gpt-4o')
        self.url = "https://openrouter.ai/api/v1/chat/completions"
        self.max_retries = 5  # Reduced from 10 to be more reasonable
        self.temperature = self.config.get('temperature', 0.0)  # Low temperature for consistent evaluation
        
        print(f"DEBUG: VLM Judge using model: {self.model_name}")
        print(f"DEBUG: VLM Judge config: {self.config}")
        
        # Load prompts
        self.prompts = self.load_prompts()
        
        # Task configuration
        self.task = self.config.get('task', 'text2svg')
        if self.task not in self.prompts:
            raise ValueError(f"Task '{self.task}' not found in prompts. Available tasks: {list(self.prompts.keys())}")
        
        # Print remaining credits
        self.update_remaining_credit()

    def load_prompts(self):
        """Load prompts from prompts.json file"""
        prompts_path = os.path.join(os.path.dirname(__file__), 'prompts.json')
        try:
            with open(prompts_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Prompts file not found at {prompts_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in prompts file at {prompts_path}")

    def update_remaining_credit(self):
        """Check remaining OpenRouter credits"""
        headers = self.get_header()
        check_credit_url = "https://openrouter.ai/api/v1/auth/key"
        
        try:
            response = requests.get(check_credit_url, headers=headers)
            response.raise_for_status()
            response_json = response.json()
            
            if "error" not in response_json:
                usage_credits = response_json["data"]["limit_remaining"]
                print(f"Remaining OpenRouter credits: {usage_credits}")
            else:
                print("Could not retrieve credit information")
        except Exception as e:
            print(f"Error checking credits: {e}")

    # def get_header(self):
    #     """Get headers for OpenRouter API requests"""
    #     return {"Authorization": f"Bearer {self.api_key}"}
    def get_header(self):
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def __encode_pil_image(self, img, data_format="png"):
        """Encode PIL image to base64"""
        im_file = BytesIO()
        img.save(im_file, format=data_format.upper())
        im_bytes = im_file.getvalue()
        im_b64 = base64.b64encode(im_bytes).decode("utf-8")
        return im_b64

    def img_url(self, img, data_format="png"):
        """Create data URL for image"""
        return f"data:image/{data_format};base64,{self.__encode_pil_image(img, data_format)}"

    def extract_score_from_response(self, response_text):
        """Extract numerical score from VLM response"""
        # Handle direct numeric responses
        if isinstance(response_text, (int, float)):
            score = float(response_text)
            if 1 <= score <= 10:
                return score
            else:
                return None
        
        # Handle string responses
        if not isinstance(response_text, str):
            return None
        
        # Check for content policy rejections
        rejection_patterns = [
            "i'm sorry, i can't assist",
            "i cannot assist",
            "i'm unable to assist",
            "i can't help with",
            "against my guidelines",
            "content policy"
        ]
        
        response_lower = response_text.lower()
        for pattern in rejection_patterns:
            if pattern in response_lower:
                print(f"Content policy rejection detected: {response_text[:100]}...")
                return None
            
        # Look for score pattern like "score: 8" or "8/10" or just "8"
        score_patterns = [
            r'score:\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)/10',
            r'rate[ds]?\s*(\d+(?:\.\d+)?)',
            r'(\d+(?:\.\d+)?)\s*out\s*of\s*10',
            r'(\d+(?:\.\d+)?)'
        ]
        
        for pattern in score_patterns:
            match = re.search(pattern, response_text.lower())
            if match:
                try:
                    score = float(match.group(1))
                    # Ensure score is in valid range
                    if 1 <= score <= 10:
                        return score
                except ValueError:
                    continue
        
        # If no valid score found, return None
        print(f"Could not extract valid score from response: {response_text}")
        return None

    def model_generate(self, url, headers, payload, retry_delay=1):
        """Generate response from OpenRouter API with improved retry logic"""
        for attempt in range(self.max_retries):
            try:
                response = requests.post(url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                
                if "error" in data:
                    print(f"API Error: {data['error']}")
                    return None
                
                response_text = data.get("choices", [{"message": {"content": ""}}])[0]["message"]["content"]
                if response_text.strip():
                    # Check for content policy rejections
                    rejection_keywords = ["sorry, i can't assist", "i cannot assist", "against my guidelines"]
                    response_lower = response_text.lower()
                    if any(keyword in response_lower for keyword in rejection_keywords):
                        print(f"Content policy rejection detected in attempt {attempt + 1}")
                        if attempt == self.max_retries - 1:
                            return None  # Give up after all retries
                        time.sleep(retry_delay)
                        continue  # Try again
                    return response_text
                    
            except requests.exceptions.JSONDecodeError:
                print(f"Attempt {attempt + 1} failed: Response is not in JSON format")
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {e}")
            
            if attempt < self.max_retries - 1:
                # Use linear backoff instead of exponential for faster retries
                sleep_time = retry_delay * (attempt + 1)
                print(f"Retrying in {sleep_time} seconds...")
                time.sleep(sleep_time)
        
        return None

    def evaluate_text2svg(self, gen_im, caption):
        """Evaluate text-to-image generation"""
        prompt_config = self.prompts['text2svg']
        
        content = [
            {"type": "text", "text": prompt_config['system_prompt']},
            {
                "type": "text", 
                "text": prompt_config['user_prompt'].format(caption=caption)
            },
            {
                "type": "image_url",
                "image_url": {"url": self.img_url(gen_im)}
            }
        ]
        
        return self._make_api_call(content)

    def evaluate_sketch2svg(self, gt_im, gen_im):
        """Evaluate sketch-to-image generation"""
        prompt_config = self.prompts['sketch2svg']
        
        content = [
            {"type": "text", "text": prompt_config['system_prompt']},
            {"type": "text", "text": prompt_config['user_prompt']},
            {
                "type": "image_url",
                "image_url": {"url": self.img_url(gt_im)}
            },
            {
                "type": "image_url",
                "image_url": {"url": self.img_url(gen_im)}
            }
        ]
        
        return self._make_api_call(content)

    def evaluate_svg_editing(self, gt_im, gen_im, caption):
        """Evaluate image editing with instructions"""
        prompt_config = self.prompts['svg_editing']
        
        content = [
            {"type": "text", "text": prompt_config['system_prompt']},
            {
                "type": "text", 
                "text": prompt_config['user_prompt'].format(caption=caption)
            },
            {
                "type": "image_url",
                "image_url": {"url": self.img_url(gt_im)}
            },
            {
                "type": "image_url",
                "image_url": {"url": self.img_url(gen_im)}
            }
        ]
        
        return self._make_api_call(content)

    def _make_api_call(self, content):
        """Make API call to OpenRouter"""
        headers = self.get_header()
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 500,
            "temperature": self.temperature,
        }
        
        response_text = self.model_generate(self.url, headers, payload)
        if response_text:
            score = self.extract_score_from_response(response_text)
            return score
        return None

    def metric(self, **kwargs):
        """Main metric method that routes to appropriate evaluation based on task"""
        if self.task == 'text2svg':
            gen_im = kwargs.get('gen_im')
            caption = kwargs.get('caption')
            if gen_im is None or caption is None:
                raise ValueError("text2svg task requires 'gen_im' and 'caption'")
            return self.evaluate_text2svg(gen_im, caption)
            
        elif self.task == 'sketch2svg':
            gt_im = kwargs.get('gt_im')
            gen_im = kwargs.get('gen_im')
            if gt_im is None or gen_im is None:
                raise ValueError("sketch2svg task requires 'gt_im' and 'gen_im'")
            return self.evaluate_sketch2svg(gt_im, gen_im)
            
        elif self.task == 'svg_editing':
            gt_im = kwargs.get('gt_im')
            gen_im = kwargs.get('gen_im')
            caption = kwargs.get('caption')
            if gt_im is None or gen_im is None or caption is None:
                raise ValueError("svg_editing task requires 'gt_im', 'gen_im', and 'caption'")
            return self.evaluate_svg_editing(gt_im, gen_im, caption)
            
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def calculate_score(self, batch, update=True):
        """Override calculate_score to handle batch processing with progress tracking"""
        values = []
        batch_size = len(next(iter(batch.values())))
        
        for index in tqdm(range(batch_size), desc=f"VLM Judge ({self.task})"):
            kwargs = {}
            for key in ["gt_im", "gen_im", "caption"]:
                if key in batch:
                    kwargs[key] = batch[key][index]
            
            try:
                measure = self.metric(**kwargs)
                if measure is not None and not math.isnan(measure):
                    values.append(measure)
                else:
                    print(f"Invalid score for sample {index}")
            except Exception as e:
                print(f"Error calculating VLM judge metric for sample {index}: {e}")
                continue

        if not values:
            print("No valid values found for VLM judge metric calculation.")
            return float("nan"), []

        score = sum(values) / len(values)
        if update:
            self.meter.update(score, len(values))
            return self.meter.avg, values
        else:
            return score, values 