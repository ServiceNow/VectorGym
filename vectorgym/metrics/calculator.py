"""Main metrics calculator for VectorGym."""

from .compute_l2 import L2DistanceCalculator
from .compute_LPIPS import LPIPSDistanceCalculator
from .compute_SSIM import SSIMDistanceCalculator
from .compute_fid import FIDCalculator
from .compute_clip_score import CLIPScoreCalculator
from .compute_dino_score import DINOScoreCalculator
from .compute_vlm_judge import VLMJudgeCalculator
from .compute_generation_time import GenerationTimeCalculator
from .count_token_length import CountTokenLength
from .util import AverageMeter
from ..utils.svg import rasterize_svg


class SVGMetrics:
    """Main metrics calculator for SVG evaluation."""
    
    def __init__(self, config=None):
        default_config = {
            'L2': True,
            'Masked-L2': False,
            'LPIPS': False,
            'SSIM': False,
            'FID': False,
            'FID_clip': False,
            'CLIPScore': False,
            'CountTokenLength': False,
            'ratio_post_processed': True,
            'ratio_non_compiling': True,
            'DinoScore': True,
            'VLMJudge': False,
            'GenerationTime': False,
        }
        
        if config is not None:
            self.config = config
        else:
            self.config = default_config
        
        # Get VLMJudge config if enabled
        vlm_judge_config = None
        if self.config.get('VLMJudge'):
            vlm_judge_config = self.config.get('VLMJudge_config', {})
        
        self.metrics = {
            'L2': L2DistanceCalculator,
            'Masked-L2': lambda: L2DistanceCalculator(masked_l2=True),
            'LPIPS': LPIPSDistanceCalculator,
            'SSIM': SSIMDistanceCalculator,
            'FID': lambda: FIDCalculator(model_name='InceptionV3'),
            'FID_clip': lambda: FIDCalculator(model_name='ViT-B/32'),
            'CLIPScore': CLIPScoreCalculator,
            'CountTokenLength': CountTokenLength,
            'ratio_post_processed': AverageMeter,
            'ratio_non_compiling': AverageMeter,
            'DinoScore': DINOScoreCalculator,
            'VLMJudge': lambda: VLMJudgeCalculator(config=vlm_judge_config),
            'GenerationTime': GenerationTimeCalculator,
        }
        
        self.active_metrics = {
            k: v() for k, v in self.metrics.items() if self.config.get(k)
        }
    
    def reset(self):
        """Reset all active metrics."""
        for metric in self.active_metrics.values():
            metric.reset()
    
    def _get_sample_id(self, json_item):
        """Return a sample identifier from a saved json/meta dict."""
        return json_item.get('outpath_filename') or json_item.get('sample_id')
    
    def batch_contains_raster(self, batch):
        """Check if batch contains rasterized images."""
        return "gt_im" in batch and "gen_im" in batch
    
    def batch_contains_svg(self, batch):
        """Check if batch contains SVG strings."""
        return "gt_svg" in batch and "gen_svg" in batch
    
    def calculate_metrics(self, batch, update=True):
        """Calculate all active metrics on a batch."""
        if not self.batch_contains_raster(batch):
            batch["gt_im"] = [rasterize_svg(svg) for svg in batch["gt_svg"]]
            batch["gen_im"] = [rasterize_svg(svg) for svg in batch["gen_svg"]]
        
        avg_results_dict = {}
        all_results_dict = {}
        
        # Initialize all_results_dict
        for i, json_item in enumerate(batch['json']):
            sample_id = self._get_sample_id(json_item)
            if sample_id is None:
                raise ValueError(
                    f"Could not find 'outpath_filename' or 'sample_id' in batch['json'][{i}]"
                )
            all_results_dict[sample_id] = {}
        
        for metric_name, metric in self.active_metrics.items():
            print(f"Calculating {metric_name}...")
            
            # Handle metrics that return both average and per-sample results
            if metric_name in [
                'L2', 'Masked-L2', 'SSIM', 'CLIPScore', 'LPIPS',
                'CountTokenLength', 'DinoScore', 'VLMJudge', 'GenerationTime'
            ]:
                avg_result, list_result = metric.calculate_score(batch, update=update)
                avg_results_dict[metric_name] = avg_result
                
                # Store individual results
                for i, result in enumerate(list_result):
                    sample_id = self._get_sample_id(batch['json'][i])
                    all_results_dict[sample_id][metric_name] = result
            
            # Handle FID metrics that only return average
            elif metric_name in ['FID', 'FID_clip']:
                avg_results_dict[metric_name] = metric.calculate_score(batch)
            
            # Handle ratio metrics
            else:
                self._handle_ratio_metric(
                    metric_name, metric, batch, avg_results_dict, all_results_dict
                )
            
            metric.reset()
        
        print("Average results: \n", avg_results_dict)
        return avg_results_dict, all_results_dict
    
    def _handle_ratio_metric(
        self, metric_name, metric, batch, avg_results_dict, all_results_dict
    ):
        """Helper method to handle ratio-based metrics."""
        metric_key = metric_name.replace('avg_', '').replace('ratio_', '')
        
        for item in batch['json']:
            sample_id = self._get_sample_id(item)
            # Handle both 'non_compiling' and 'no_compile' for backward compatibility
            if metric_key == 'non_compiling' and metric_key not in item:
                value = 1 if item.get('no_compile', False) else 0
            else:
                value = item.get(metric_key, 0)
            all_results_dict[sample_id][metric_name] = value
            metric.update(value, 1)
        
        avg_results_dict[metric_name] = metric.avg

