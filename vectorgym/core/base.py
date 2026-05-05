"""Base validator abstract class."""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from omegaconf import OmegaConf
from pathlib import Path
import os
import json
from datetime import datetime


class BaseValidator(ABC):
    """Abstract base class for all SVG validators."""
    
    def __init__(self, config: OmegaConf):
        """
        Initialize validator with configuration.
        
        Args:
            config: OmegaConf configuration object
        """
        self.config = config
        self.task = config.model.task
        self.model_name = config.model.name
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
        
        # Initialize results storage
        self.results: Dict[str, Any] = {}
        self.table_data: Dict[str, tuple] = {}
        
        # Initialize wandb if needed
        if self.report_to_wandb:
            self._init_wandb(config)
        
        # Initialize metrics (will be set by subclasses or base)
        self.metrics = None
    
    @abstractmethod
    def get_out_dir(self, config: OmegaConf) -> str:
        """
        Get output directory for this validator.
        
        Args:
            config: Configuration object
            
        Returns:
            Base output directory path
        """
        pass
    
    @abstractmethod
    def generate_svg(self, batch: Dict[str, Any], generate_config: Dict[str, Any]) -> List[str]:
        """
        Generate SVG from batch data.
        
        Args:
            batch: Batch of input data
            generate_config: Generation parameters
            
        Returns:
            List of generated SVG strings
        """
        pass
    
    @abstractmethod
    def postprocess_svg(self, text: str, filename: str = "unknown") -> Dict[str, Any]:
        """
        Post-process generated SVG text.
        
        Args:
            text: Generated SVG text
            filename: Optional filename for logging
            
        Returns:
            Dictionary with keys: svg, svg_raw, post_processed, no_compile
        """
        pass
    
    def release_memory(self):
        """Release any held resources (model memory, etc.)."""
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

