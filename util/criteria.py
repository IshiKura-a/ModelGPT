from abc import ABC, abstractmethod
from typing import Any, Dict, List

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.functional import auroc


class Metric(ABC):
    @staticmethod
    def kl_div(predictions: Tensor, target: Tensor, t: float = 1.0, reduction: str = 'sum'):
        return F.kl_div(F.log_softmax(predictions / t, dim=-1), (target / t).softmax(dim=-1).clip(min=1e-8),
                        reduction=reduction) * (t * t)

    @staticmethod
    def js_div(predictions: Tensor, target: Tensor, t: float, reduction: str = 'sum'):
        soft_target = target / t
        mid = (predictions.softmax(dim=-1) + soft_target.softmax(dim=-1)) / 2
        return F.kl_div(F.log_softmax(predictions, dim=-1), mid, reduction=reduction) + \
            F.kl_div(F.log_softmax(soft_target, dim=-1), mid, reduction=reduction)

    def __init__(self):
        self._name = 'Metric'
        self._loss = 0
        self._cnt = 0

    @abstractmethod
    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        raise NotImplementedError()

    def get_results(self) -> Any:
        return self._loss / self._cnt if self._cnt != 0 else 0

    def reset(self):
        self._loss = 0
        self._cnt = 0

    def __str__(self):
        return f'{self.name} = {self.get_results():.3f}'

    @property
    def name(self):
        return self._name


class Accuracy(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'Acc'
        self._correct = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = predictions.softmax(dim=-1).argmax(dim=-1)
        self._cnt += target.shape[0]
        self._correct += (predictions.long() == target.long()).sum()
        return self

    def get_results(self) -> float:
        return (self._correct / self._cnt).item() if self._cnt != 0 else 0

    def reset(self):
        self._cnt = 0
        self._correct = 0


class BinaryAccuracy(Accuracy):
    def __init__(self):
        super().__init__()
        self._name = 'BAcc'
        self._correct = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = (predictions >= 0.5)
        self._cnt += target.shape[0]
        self._correct += (predictions.long() == target.long()).sum()
        return self


class TopKAccuracy(Accuracy):
    def __init__(self, k: int):
        super().__init__()
        self._name = f'Acc@{k}'
        self._correct = 0
        self._cnt = 0
        self._k = k

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        predictions = predictions.softmax(dim=-1)
        predictions = torch.topk(predictions, k=self._k).indices
        self._cnt += target.shape[0]
        for i in range(self._k):
            self._correct += (predictions[:, i].long() == target.long()).sum()
        return self


class CrossEntropy(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'CE'
        self._ce = nn.CrossEntropyLoss(reduction='sum')
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._ce(predictions, target)
        return self


class KLDivLoss(Metric):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'KLDivLoss'
        self._fn = lambda x, y: self.kl_div(x, y, t)
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._fn(predictions, target)
        return self


class JSDivLoss(KLDivLoss):
    def __init__(self, t: float = 1.0):
        super().__init__()
        self._name = 'JSDivLoss'
        self._fn = lambda x, y: self.js_div(x, y, t)

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._fn(predictions, target)
        return self


class BCELoss(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'BCE'
        self._bce = nn.BCEWithLogitsLoss(reduction='sum')
        self._loss = 0
        self._cnt = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._cnt += predictions.shape[0]
        self._loss += self._bce(predictions, target.float().unsqueeze(-1))
        return self


class MSELoss(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'MSE'
        self._mse = nn.MSELoss(reduction='sum')
        self._loss = 0

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._loss += self._mse(predictions.squeeze(), target)
        self._cnt += predictions.shape[0]
        return self


class Compose(Metric):
    def __init__(self, metric: List[Metric] = []):
        super().__init__()
        self._list = metric
        self.reset()

    def __getitem__(self, index: int) -> Metric:
        return self._list[index]

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        for metric in self._list:
            metric.update(predictions, target, **kwargs)
        return self

    def get_results(self) -> Dict[str, float]:
        return {
            metric.name: metric.get_results() for metric in self._list
        }

    def reset(self):
        for metric in self._list:
            metric.reset()

    @staticmethod
    def compose(metrics: List[Metric]) -> 'Compose':
        _list = []
        for metric in metrics:
            if isinstance(metric, Compose):
                _list += metric._list
            else:
                _list.append(metric)
        return Compose(_list)

    def __str__(self):
        return ' '.join([metric.__str__() for metric in self._list])


class AUROC(Metric):
    def __init__(self):
        super().__init__()
        self._name = 'AUROC'

    def update(self, predictions: Tensor, target: Tensor, **kwargs) -> 'Metric':
        self._loss += auroc(predictions.softmax(dim=-1), target)
        self._cnt += predictions.shape[0]
        return self
