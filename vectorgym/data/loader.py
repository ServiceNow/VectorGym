"""Unified data loader for all tasks."""

from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from .processor import prepare_task_data


def get_unified_dataloader(config: OmegaConf) -> DataLoader:
    """
    Create unified dataloader that works for all tasks.
    
    Args:
        config: Configuration object
        
    Returns:
        DataLoader instance
    """
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

