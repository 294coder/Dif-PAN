# author: Zihan
# date: 2023/11/12

from dataclasses import dataclass
import numpy as np

@dataclass()
class BestMetricSaveChecker:
    _best_metric: float
    metric_name: str
    check_order: str
    
    def __init__(self, metric_name: str, check_order: str='up'):
        self.metric_name = metric_name
        assert check_order in ['up', 'down']
        self.check_order = check_order
        if check_order == 'up': default_best_metric_val = -np.Inf
        else: default_best_metric_val = np.Inf

        self._best_metric = default_best_metric_val
        self._check_fn = lambda new, old: new > old if check_order=='up' else lambda old, new: new < old
        
    def __call__(self, val_metrics: dict[str, float]):
        assert self.metric_name in val_metrics.keys(), f'@metric_name {self.metric_name} should in @val_metrics, but got {val_metrics}'
        best_val = val_metrics[self.metric_name]
        prev_val = self._best_metric
        
        _save = self._check_fn(best_val, prev_val)
        if _save: self._best_metric = best_val
        
        return _save
    
    @property
    def best_metric(self):
        return self._best_metric
    
    
if __name__ == '__main__':
    checker = BestMetricSaveChecker('ssim')
    
    val_d1 = {'psnr': 10, 'ssim':0.8}
    val_d2 = {'psnr': 20, 'ssim':0.9}
    
    print(checker.best_metric)
    
    checker(val_d1)
    print(checker.best_metric)
    
    checker(val_d2)
    print(checker.best_metric)