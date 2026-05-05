"""Base validator class with common functionality for all validators."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm
from PIL import Image
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from datasets import load_dataset
from torch.utils.data import DataLoader
from svgpathtools import svgstr2paths
import math
import matplotlib.pyplot as plt
import textwrap
from io import BytesIO

# Import utilities (will be created)
from ..utils.svg import robust_svg_to_pil, clean_svg, use_placeholder, get_svg_original_size
from ..data.processor import prepare_task_data
from ..core.registry import register_validator


class BaseValidator(ABC):
    """Base class for all SVG validators with common functionality."""
    
    def __init__(self, config: OmegaConf):
        """Initialize validator with configuration."""
        self.config = config
        self.task = config.task
        self.backend = config.backend
        self.model_name = config.model_name
        self.report_to_wandb = config.run.report_to == 'wandb'
        
        # Create output directory
        date_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir_base = self.get_out_dir(config)
        self.out_dir = os.path.join(
            out_dir_base,
            f'results_dataset_{config.dataset.dataset_name.split("/")[-1]}_{date_time}'
        )
        config['run']['results_dir'] = self.out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Save config
        config_path = os.path.join(self.out_dir, "config.yaml")
        with open(config_path, "w") as f:
            OmegaConf.save(config=config, f=f)
        
        # Initialize storage
        self.results: Dict[str, Any] = {}
        self.table_data: Dict[str, tuple] = {}
        
        # Initialize wandb if needed
        if self.report_to_wandb:
            self._init_wandb(config)
        
        # Initialize metrics (imported later to avoid circular deps)
        from ..metrics.calculator import SVGMetrics
        self.metrics = SVGMetrics(config.metrics)
        
        # Initialize dataloader
        self.dataloader = self.get_unified_dataloader(config)
    
    @abstractmethod
    def get_out_dir(self, config: OmegaConf) -> str:
        """Get output directory for this validator."""
        pass
    
    @abstractmethod
    def generate_svg(self, batch: Dict[str, Any], generate_config: Dict[str, Any]) -> List[str]:
        """Generate SVG from batch data."""
        pass
    
    @abstractmethod
    def postprocess_svg(self, text: str, filename: str = "unknown") -> Dict[str, Any]:
        """Post-process generated SVG text."""
        pass
    
    def release_memory(self):
        """Release any held resources."""
        pass
    
    def _init_wandb(self, config: OmegaConf):
        """Initialize wandb logging."""
        try:
            import wandb
            run_id = f"{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=config.run.project_name,
                name=run_id,
                config=OmegaConf.to_container(config, resolve=True)
            )
            self.results_table = wandb.Table(columns=[
                "sample_id", "svg", "svg_raw", "svg_gt",
                "no_compile", "post_processed",
                "original_image", "generated_image", "comparison_image"
            ])
        except Exception as e:
            print(f"Failed to initialize wandb: {e}")
    
    def get_unified_dataloader(self, config: OmegaConf) -> DataLoader:
        """Unified dataloader that works for all tasks."""
        data = prepare_task_data(config)
        
        def simple_collate(batch):
            keys = batch[0].keys()
            return {k: [b[k] for b in batch] for k in keys}
        
        return DataLoader(
            data,
            batch_size=config.dataset.batch_size,
            shuffle=False,
            num_workers=config.dataset.num_workers,
            collate_fn=simple_collate
        )
    
    def rasterize(self, pred_svg: str, gt_svg: str, RASTERIZE_SIZE: int = 384) -> tuple:
        """Rasterize SVG to PIL Image."""
        try:
            svg_raster, _ = robust_svg_to_pil(
                pred_svg,
                output_width=RASTERIZE_SIZE,
                output_height=RASTERIZE_SIZE
            )
        except Exception as e:
            print(f"Warning: Failed to rasterize generated SVG: {e}")
            svg_raster = Image.new('RGB', (RASTERIZE_SIZE, RASTERIZE_SIZE), (255, 255, 255))
        
        try:
            gt_svg_raster, _ = robust_svg_to_pil(
                gt_svg,
                output_width=RASTERIZE_SIZE,
                output_height=RASTERIZE_SIZE
            )
        except Exception as e:
            print(f"Warning: Failed to rasterize ground-truth SVG: {e}")
            gt_svg_raster = Image.new('RGB', (RASTERIZE_SIZE, RASTERIZE_SIZE), (255, 255, 255))
        
        return svg_raster, gt_svg_raster
    
    def calculate_mse(self, pred_im: Image.Image, gt_im: Image.Image) -> float:
        """Calculate MSE between two images."""
        image1_tensor = ToTensor()(gt_im)
        image2_tensor = ToTensor()(pred_im)
        mse = F.mse_loss(image1_tensor, image2_tensor)
        return float(mse.item())
    
    def post_process_svg(self, text: str) -> Dict[str, Any]:
        """Post-process a single SVG text (default implementation)."""
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
    
    def generate_and_process_batch(
        self,
        batch: Dict[str, Any],
        generate_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Generate and post-process SVGs for a batch."""
        generated_outputs = self.generate_svg(batch, generate_config)
        processed_results = [self.postprocess_svg(output) for output in generated_outputs]
        
        # Attach generation metadata if available
        gen_meta = getattr(self, 'last_generation_meta', None)
        if gen_meta is not None:
            for idx, res in enumerate(processed_results):
                if idx < len(gen_meta):
                    res_meta = gen_meta[idx] or {}
                    res['gen_time'] = float(res_meta.get('time', res_meta.get('duration', 0.0)))
                    res['token_count'] = int(res_meta.get('token_count', 0))
                else:
                    res['gen_time'] = 0.0
                    res['token_count'] = 0
        
        # Handle multiple generations
        num_generations = generate_config.get('num_generations', 1)
        if num_generations > 1:
            self.save_all_generations(batch, processed_results, num_generations)
            best_results = self.select_best_output(batch, processed_results, num_generations)
        else:
            best_results = processed_results
        
        return best_results
    
    def select_best_output(
        self,
        batch: Dict[str, Any],
        processed_results: List[Dict[str, Any]],
        n: int
    ) -> List[Dict[str, Any]]:
        """Select the best output from multiple generations."""
        best_outputs = []
        svg_field = 'Svg' if 'Svg' in batch else 'original_svg'
        
        for i, gt_svg in enumerate(batch[svg_field]):
            mses = []
            has_content = []
            
            for j in range(n):
                result = processed_results[i * n + j]
                pred_im, gt_im = self.rasterize(
                    result['svg'],
                    gt_svg,
                    RASTERIZE_SIZE=self.config.dataset.render_size
                )
                mse = self.calculate_mse(pred_im, gt_im)
                mses.append(mse)
                
                svg_content = result['svg'].strip()
                has_meaningful_content = (
                    not result.get('no_compile', False) and
                    len(svg_content) > 100 and
                    any(tag in svg_content for tag in ['<circle', '<rect', '<path', '<line', '<ellipse', '<polygon'])
                )
                has_content.append(has_meaningful_content)
            
            # Prefer generations with content
            content_indices = [j for j, has_c in enumerate(has_content) if has_c]
            if content_indices:
                content_mses = [mses[j] for j in content_indices]
                best_content_idx = content_indices[np.argmin(content_mses)]
                best_outputs.append(processed_results[i * n + best_content_idx])
            else:
                best_output_index = np.argmin(mses)
                best_outputs.append(processed_results[i * n + best_output_index])
        
        return best_outputs
    
    def save_all_generations(
        self,
        batch: Dict[str, Any],
        processed_results: List[Dict[str, Any]],
        num_generations: int
    ):
        """Save all generated outputs for each sample."""
        svg_field = 'Svg' if 'Svg' in batch else 'original_svg'
        filename_field = 'Filename' if 'Filename' in batch else 'svg_id'
        batch_size = len(batch[svg_field])
        
        for i in range(batch_size):
            sample_id = str(batch[filename_field][i]).split('.')[0]
            gt_svg = batch[svg_field][i]
            
            sample_dir = os.path.join(self.out_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            generations_dir = os.path.join(sample_dir, 'generations')
            os.makedirs(generations_dir, exist_ok=True)
            
            # Save ground truth
            with open(os.path.join(sample_dir, f"{sample_id}_gt.svg"), 'w', encoding='utf-8') as f:
                f.write(gt_svg)
            
            # Save input data
            if 'prompt' in batch:
                with open(os.path.join(sample_dir, f"{sample_id}_input_prompt.txt"), 'w', encoding='utf-8') as f:
                    f.write(batch['prompt'][i])
            
            if 'images' in batch and batch['images'][i] is not None:
                batch['images'][i].save(os.path.join(sample_dir, f"{sample_id}_input_sketch.png"))
            
            # Save all generations
            generation_metadata = []
            for gen_idx in range(num_generations):
                result_idx = i * num_generations + gen_idx
                if result_idx < len(processed_results):
                    res = processed_results[result_idx]
                    gen_filename = f"{sample_id}_gen_{gen_idx:02d}"
                    
                    with open(os.path.join(generations_dir, f"{gen_filename}.svg"), 'w', encoding='utf-8') as f:
                        f.write(res['svg'])
                    with open(os.path.join(generations_dir, f"{gen_filename}_raw.svg"), 'w', encoding='utf-8') as f:
                        f.write(res['svg_raw'])
                    
                    try:
                        svg_raster, _ = self.rasterize(
                            res['svg'],
                            gt_svg,
                            RASTERIZE_SIZE=self.config.dataset.render_size
                        )
                        svg_raster.save(os.path.join(generations_dir, f"{gen_filename}.png"))
                        gt_raster, _ = self.rasterize(
                            gt_svg,
                            gt_svg,
                            RASTERIZE_SIZE=self.config.dataset.render_size
                        )
                        mse = self.calculate_mse(svg_raster, gt_raster)
                    except Exception as e:
                        print(f"Error rasterizing generation {gen_idx} for sample {sample_id}: {e}")
                        mse = float('inf')
                    
                    generation_metadata.append({
                        'generation_id': gen_idx,
                        'mse': float(mse),
                        'post_processed': res['post_processed'],
                        'no_compile': res['no_compile'],
                        'svg_length': len(res['svg']),
                        'raw_svg_length': len(res['svg_raw']),
                        'gen_time': res.get('gen_time', 0.0),
                        'token_count': res.get('token_count', 0)
                    })
            
            # Save metadata
            metadata_path = os.path.join(sample_dir, 'all_generations_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump({
                    'sample_id': sample_id,
                    'total_generations': num_generations,
                    'generations': generation_metadata
                }, f, indent=4, sort_keys=True)
    
    def save_results(self, results: List[Dict[str, Any]], batch: Dict[str, Any], batch_idx: int):
        """Save results from generation."""
        svg_field = 'Svg' if 'Svg' in batch else 'original_svg'
        filename_field = 'Filename' if 'Filename' in batch else 'svg_id'
        
        for i, sample in enumerate(batch[svg_field]):
            sample_id = str(batch[filename_field][i]).split('.')[0]
            res = results[i]
            res['sample_id'] = sample_id
            res['gt_svg'] = sample
            
            sample_dir = os.path.join(self.out_dir, sample_id)
            os.makedirs(sample_dir, exist_ok=True)
            
            # Store input data
            if 'prompt' in batch:
                res['input_prompt'] = batch['prompt'][i]
                with open(os.path.join(sample_dir, f"{sample_id}_input_prompt.txt"), 'w', encoding='utf-8') as f:
                    f.write(batch['prompt'][i])
            
            if 'images' in batch and batch['images'][i] is not None:
                res['input_sketch'] = batch['images'][i]
                batch['images'][i].save(os.path.join(sample_dir, f"{sample_id}_input_sketch.png"))
            
            # Save SVG files and rasterized images
            svg_raster, gt_svg_raster = self._save_svg_files(sample_dir, sample_id, res)
            
            # Save metadata
            metadata = {k: v for k, v in res.items() if k != 'input_sketch'}
            with open(os.path.join(sample_dir, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=4, sort_keys=True)
            
            res['gen_im'] = svg_raster
            res['gt_im'] = gt_svg_raster
            self.results[sample_id] = res
            
            # Update wandb table if enabled
            if self.report_to_wandb and self.config.run.log_images:
                self._update_wandb_table(sample_id, res, gt_svg_raster, svg_raster)
    
    def _save_svg_files(
        self,
        sample_dir: str,
        sample_id: str,
        res: Dict[str, Any]
    ) -> tuple:
        """Save SVG files and rasterized images."""
        with open(os.path.join(sample_dir, f"{sample_id}.svg"), 'w', encoding='utf-8') as f:
            f.write(res['svg'])
        with open(os.path.join(sample_dir, f"{sample_id}_raw.svg"), 'w', encoding='utf-8') as f:
            f.write(res['svg_raw'])
        with open(os.path.join(sample_dir, f"{sample_id}_gt.svg"), 'w', encoding='utf-8') as f:
            f.write(res['gt_svg'])
        
        svg_raster, gt_svg_raster = self.rasterize(
            res['svg'],
            res['gt_svg'],
            RASTERIZE_SIZE=self.config.dataset.render_size
        )
        
        svg_raster.save(os.path.join(sample_dir, f"{sample_id}_generated.png"))
        gt_svg_raster.save(os.path.join(sample_dir, f"{sample_id}_original.png"))
        
        return svg_raster, gt_svg_raster
    
    def _update_wandb_table(
        self,
        sample_id: str,
        res: Dict[str, Any],
        gt_im: Image.Image,
        gen_im: Image.Image
    ):
        """Update wandb table with sample results."""
        try:
            import wandb
            row = (
                sample_id,
                res['svg'],
                res['svg_raw'],
                res['gt_svg'],
                res['no_compile'],
                res['post_processed'],
                wandb.Image(gt_im),
                wandb.Image(gen_im),
                None  # Placeholder for comparison_image
            )
            self.table_data[sample_id] = row
            self._log_wandb_table()
        except Exception as e:
            print(f"Failed to update wandb table: {e}")
    
    def _log_wandb_table(self):
        """Log wandb table."""
        if not self.report_to_wandb:
            return
        try:
            import wandb
            table = wandb.Table(columns=[
                "sample_id", "svg", "svg_raw", "svg_gt",
                "no_compile", "post_processed",
                "original_image", "generated_image", "comparison_image"
            ])
            for row in self.table_data.values():
                table.add_data(*row)
            wandb.log({"results_table": table})
            self.results_table = table
        except Exception as e:
            print(f"Failed to log wandb table: {e}")
    
    def validate(self):
        """Main validation loop."""
        for i, batch in enumerate(tqdm(self.dataloader, desc="Validating")):
            if self.config.generation_params.get('generation_sweep', False):
                results = self.run_temperature_sweep(batch)
            else:
                results = self.generate_and_process_batch(batch, self.config.generation_params)
            
            self.save_results(results, batch, i)
        
        self.release_memory()
        self.calculate_and_save_metrics()
        
        # Final wandb logging
        if self.report_to_wandb and self.config.run.log_images:
            try:
                import wandb
                wandb.log({"results_table": self.results_table})
            except Exception as e:
                print(f"Failed to log final results table to wandb: {e}")
    
    def preprocess_results(self) -> Dict[str, List]:
        """Preprocess results from self.results into batch format."""
        batch = {
            'gen_svg': [],
            'gt_svg': [],
            'gen_im': [],
            'gt_im': [],
            'json': [],
            'caption': []
        }
        
        for sample_id, result_dict in self.results.items():
            if self.config.generation_params.get('generation_sweep', False):
                result = result_dict[list(result_dict.keys())[0]]
            else:
                result = result_dict
            
            batch['gen_svg'].append(result['svg'])
            batch['gt_svg'].append(result['gt_svg'])
            batch['gen_im'].append(result['gen_im'])
            batch['gt_im'].append(result['gt_im'])
            batch['json'].append(result)
            
            # Extract caption from input_prompt
            caption = ""
            if 'input_prompt' in result:
                prompt = result['input_prompt']
                if "Convert the input text into an SVG: " in prompt:
                    caption = prompt.split("Convert the input text into an SVG: ", 1)[1]
                elif "Edit this SVG with the following instruction: " in prompt:
                    caption = prompt.split("Edit this SVG with the following instruction: ", 1)[1].split("\n\nOriginal SVG:")[0]
                else:
                    caption = prompt
            
            batch['caption'].append(caption)
        
        return batch
    
    def calculate_and_save_metrics(self):
        """Calculate and save metrics."""
        batch_results = self.preprocess_results()
        avg_results, all_results = self.metrics.calculate_metrics(batch_results)
        
        out_path_results = os.path.join(self.out_dir, 'results')
        os.makedirs(out_path_results, exist_ok=True)
        
        # Save average results
        with open(os.path.join(out_path_results, 'results_avg.json'), 'w') as f:
            json.dump(avg_results, f, indent=4, sort_keys=True)
        
        # Save detailed results
        df = pd.DataFrame.from_dict(all_results, orient='index')
        df.to_csv(os.path.join(out_path_results, 'all_results.csv'))
        
        # Log to wandb
        if self.report_to_wandb:
            try:
                import wandb
                wandb.log({'avg_metrics': avg_results})
            except Exception as e:
                print(f"Error logging average metrics to wandb: {e}")
        
        # Create comparison plots
        self.create_comparison_plots_with_metrics(all_results, max_samples=100)
        
        # Create generations summary if multiple generations were used
        if hasattr(self.config.generation_params, 'num_generations') and self.config.generation_params.num_generations > 1:
            self.create_generations_summary()
    
    def create_comparison_plots_with_metrics(self, all_metrics: Dict[str, Any], max_samples: int = 100):
        """Create and save comparison plots with metrics."""
        counter = 0
        for sample_id, metrics in all_metrics.items():
            if sample_id not in self.results:
                continue
            counter += 1
            
            if counter > max_samples:
                break
            
            result = self.results[sample_id]
            sample_dir = os.path.join(self.out_dir, sample_id)
            
            gt_raster = result.get('gt_im')
            gen_raster = result.get('gen_im')
            if gt_raster is None or gen_raster is None:
                continue
            
            output_path = os.path.join(sample_dir, f"{sample_id}_comparison.png")
            comp_img = self.create_comparison_plot(sample_id, gt_raster, gen_raster, metrics, output_path, result)
            result['comparison_image'] = comp_img
            
            # Update wandb table
            if self.report_to_wandb and sample_id in self.table_data and self.config.run.log_images:
                import wandb
                row = list(self.table_data[sample_id])
                row[-1] = wandb.Image(comp_img)
                self.table_data[sample_id] = tuple(row)
                self._log_wandb_table()
        
        # Create grid visualization
        self._create_comparison_grid(max_samples)
    
    def create_comparison_plot(
        self,
        sample_id: str,
        gt_raster: Image.Image,
        gen_raster: Image.Image,
        metrics: Dict[str, Any],
        output_path: str,
        result_data: Optional[Dict[str, Any]] = None
    ) -> Image.Image:
        """Create comparison plot showing input, GT, and generated SVG."""
        has_input_data = result_data and ('input_prompt' in result_data or 'input_sketch' in result_data)
        
        if has_input_data:
            fig, (ax_metrics, ax_input, ax_images) = plt.subplots(
                3, 1, figsize=(12, 12),
                gridspec_kw={'height_ratios': [1, 2, 4]}
            )
        else:
            fig, (ax_metrics, ax_images) = plt.subplots(
                2, 1, figsize=(12, 8),
                gridspec_kw={'height_ratios': [1, 4]}
            )
        
        fig.suptitle(f'Generation Results for {sample_id}', fontsize=16)
        
        # Metrics text
        if metrics:
            metrics_text = "Metrics:\n"
            for key, val in metrics.items():
                if isinstance(val, list) and val:
                    metrics_text += f"{key}: {val[-1]:.4f}\n"
                elif isinstance(val, (int, float)):
                    metrics_text += f"{key}: {val:.4f}\n"
                else:
                    metrics_text += f"{key}: {val}\n"
        else:
            metrics_text = "No metrics available."
        
        ax_metrics.text(0.5, 0.5, metrics_text, fontfamily='monospace',
                       horizontalalignment='center', verticalalignment='center')
        ax_metrics.axis('off')
        
        # Input visualization
        if has_input_data:
            input_prompt = result_data.get('input_prompt')
            input_sketch = result_data.get('input_sketch')
            
            if input_sketch is not None and input_prompt:
                ax_input.set_title('Input: Sketch Image + Prompt')
                sketch_array = np.array(input_sketch)
                wrapped_prompt = textwrap.fill(input_prompt, width=60)
                ax_input.text(0.75, 0.5, wrapped_prompt, fontsize=10, ha='left', va='center',
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"),
                             transform=ax_input.transAxes)
                ax_input.imshow(sketch_array, extent=[0, 0.6, 0, 1])
                ax_input.set_xlim(0, 1)
                ax_input.set_ylim(0, 1)
            elif input_sketch is not None:
                ax_input.set_title('Input: Sketch Image')
                ax_input.imshow(np.array(input_sketch))
            elif input_prompt:
                ax_input.set_title('Input: Text Prompt')
                wrapped_prompt = textwrap.fill(input_prompt, width=80)
                ax_input.text(0.5, 0.5, wrapped_prompt, fontsize=12, ha='center', va='center',
                             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue"),
                             transform=ax_input.transAxes)
            ax_input.axis('off')
        
        # SVG comparison
        ax_images.set_title('Ground Truth (left) vs Generated (right)')
        gt_array = np.array(gt_raster)
        gen_array = np.array(gen_raster)
        combined = np.hstack((gt_array, gen_array))
        ax_images.imshow(combined)
        ax_images.axis('off')
        
        # Save
        buf = BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.savefig(output_path, format='png', bbox_inches='tight', dpi=300)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf)
    
    def _create_comparison_grid(self, max_samples: int):
        """Create grid visualization of all samples."""
        results_dir = os.path.join(self.out_dir, 'results')
        os.makedirs(results_dir, exist_ok=True)
        
        samples = list(self.results.items())
        num_samples = min(max_samples, len(samples))
        
        if not samples:
            return
        
        grid_size = math.ceil(math.sqrt(num_samples))
        rows = cols = grid_size
        fig_size = min(20, max(8, 2 * grid_size))
        fig, axes = plt.subplots(rows, cols, figsize=(fig_size, fig_size))
        fig.suptitle(f"Comparison Grid ({num_samples} samples)", fontsize=12)
        fig.tight_layout(pad=3.0)
        
        if num_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, (sample_id, result) in enumerate(samples):
            if i >= len(axes):
                break
            gen_im = result.get('gen_im')
            gt_im = result.get('gt_im')
            
            if gen_im is None or gt_im is None:
                continue
            
            combined = np.hstack((np.array(gt_im), np.array(gen_im)))
            axes[i].imshow(combined)
            axes[i].set_title(sample_id, fontsize=8)
            axes[i].axis('off')
        
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
        
        grid_path = os.path.join(results_dir, f'comparison_grid_{num_samples}.png')
        plt.savefig(grid_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        if self.report_to_wandb and os.path.exists(grid_path):
            try:
                import wandb
                grid_image = Image.open(grid_path)
                wandb.log({"comparison_grid": wandb.Image(grid_image)})
            except Exception as e:
                print(f"Failed to log grid image to wandb: {e}")
    
    def create_generations_summary(self):
        """Create summary of all generations across samples."""
        if not os.path.exists(self.out_dir):
            return
        
        summary_data = []
        for item in os.listdir(self.out_dir):
            item_path = os.path.join(self.out_dir, item)
            if os.path.isdir(item_path) and item != 'results':
                metadata_path = os.path.join(item_path, 'all_generations_metadata.json')
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if metadata['generations']:
                        best_gen = min(metadata['generations'], key=lambda x: x['mse'])
                        worst_gen = max(metadata['generations'], key=lambda x: x['mse'])
                        avg_mse = np.mean([g['mse'] for g in metadata['generations']])
                        
                        summary_data.append({
                            'sample_id': metadata['sample_id'],
                            'total_generations': metadata['total_generations'],
                            'best_generation_id': best_gen['generation_id'],
                            'best_mse': best_gen['mse'],
                            'worst_generation_id': worst_gen['generation_id'],
                            'worst_mse': worst_gen['mse'],
                            'average_mse': avg_mse,
                            'mse_std': np.std([g['mse'] for g in metadata['generations']]),
                            'best_post_processed': best_gen['post_processed'],
                            'best_no_compile': best_gen['no_compile'],
                        })
        
        if summary_data:
            results_dir = os.path.join(self.out_dir, 'results')
            os.makedirs(results_dir, exist_ok=True)
            
            df = pd.DataFrame(summary_data)
            df.to_csv(os.path.join(results_dir, 'generations_summary.csv'), index=False)
            
            with open(os.path.join(results_dir, 'generations_summary.json'), 'w') as f:
                json.dump(summary_data, f, indent=4, sort_keys=True)
    
    def run_temperature_sweep(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run generation with temperature sweep (placeholder)."""
        # This would be implemented for temperature sweeps
        # For now, just use default generation
        return self.generate_and_process_batch(batch, self.config.generation_params)

