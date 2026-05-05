from .base_metric import BaseMetric
from .util import AverageMeter
from tqdm import tqdm


class GenerationTimeCalculator(BaseMetric):
    def __init__(self, config=None, device='cuda'):
        super().__init__()
        self.class_name = self.__class__.__name__
        self.meter = AverageMeter()

    def calculate_score(self, batch, update=True):
        """Calculate generation time metrics from batch metadata"""
        json_data = batch.get('json', [])
        values = []
        
        for item in tqdm(json_data, desc="Processing generation times"):
            gen_time = item.get('gen_time', 0.0)
            if gen_time is not None and gen_time > 0:
                values.append(gen_time)
            else:
                # Fallback for missing timing data
                values.append(0.0)
        
        if not values:
            print("No timing data found for generation time metric calculation.")
            return float("nan"), []

        avg_time = sum(values) / len(values)
        if update:
            self.meter.update(avg_time, len(values))
            return self.meter.avg, values
        else:
            return avg_time, values

    def reset(self):
        self.meter.reset()
