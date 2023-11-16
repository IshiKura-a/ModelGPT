from functools import reduce
from typing import Callable, List, Dict, OrderedDict, Tuple, Any
from abc import abstractmethod

import torch
from torch import nn, Tensor
from torch.nn.utils import spectral_norm
from torchvision.models import ResNet
from transformers import DistilBertModel


class HyperNetwork(nn.Module):
    def __init__(self, encoder: nn.Module, embedding_size: int, hidden_dim: int,
                 target_parameter: Dict[str, Any], split_model: bool = False):
        super().__init__()
        self.embedding_size = embedding_size
        self.hidden_dim = hidden_dim
        self.encoder = encoder
        self.target_parameter = target_parameter

        self.split_model = split_model

    @abstractmethod
    def forward(self, inputs: Tensor) -> Dict:
        raise NotImplementedError()


class TinyEncoder(nn.Module):
    def __init__(self, ):
        super().__init__()


class DistilBertFCHYN(HyperNetwork):
    def __init__(self,
                 encoder: DistilBertModel,
                 embedding_size: int,
                 hidden_dim: int,
                 target_parameter: OrderedDict[str, torch.Size],
                 fc_info: Tuple[str, int],
                 ):
        super().__init__(encoder, embedding_size, hidden_dim, target_parameter)

        self.embed = nn.Embedding(5, embedding_size)
        hidden_size = [hidden_dim, hidden_dim, hidden_dim]

        layers = []
        prev_size = embedding_size * 2
        for h in hidden_size:
            layers.extend([
                nn.LeakyReLU(inplace=True),
                spectral_norm(nn.Linear(prev_size, h))
            ])
            prev_size = h
        layers.append(nn.LeakyReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)
        self.target_parameter = target_parameter

        self.param_map = nn.ModuleDict({
            k.replace('.', '#'): nn.Linear(self.hidden_dim, reduce(lambda x, y: x * y, list(v))) for k, v in
            target_parameter.items()
        })

        self.fc_dim = fc_info[1]
        self.fc_name = fc_info[0]
        self.fc_w = nn.Linear(self.hidden_dim, fc_info[1] * 4)
        self.fc_b = nn.Linear(self.hidden_dim, 4)

    def forward(self, **kwargs) -> Dict:
        num_labels = kwargs['num_labels']
        del kwargs['num_labels']
        x = self.encoder(**kwargs)[0][:, 0]
        num_labels = torch.tensor(num_labels).long().to(x.device)
        label_embedding = self.embed(num_labels)
        x = self.mlp(torch.cat([x.squeeze(), label_embedding]))

        weights = {}
        for k, v in self.target_parameter.items():
            weights[k] = self.param_map[k.replace('.', '#')](x).view(v)

        i = 0 if num_labels == 0 else slice(1, 1 + num_labels, 1)
        weights[f'{self.fc_name}.weight'] = self.fc_w(x).view(4, -1)[i, :]
        weights[f'{self.fc_name}.bias'] = self.fc_b(x)[i]
        return weights


class ParameterGenerationBlock(nn.Module):
    def __init__(self,
                 hidden_dim: int,
                 target_parameter: Dict[str, torch.Size]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.target_parameter = target_parameter
        self.param_map = nn.ModuleDict({
            k.replace('.', '#'): nn.Linear(hidden_dim, reduce(lambda x, y: x * y, list(v))) for k, v in
            target_parameter.items()
        })

    def forward(self, inputs: Tensor) -> Dict:
        return {
            k: self.param_map[k.replace('.', '#')](inputs).view(v) for k, v in self.target_parameter.items()
        }


class MultiHeadLMFCHYN(HyperNetwork):
    def __init__(self,
                 encoder: DistilBertModel,
                 embedding_size: int,
                 hidden_dim: int,
                 target_parameter: Dict[str, Dict[str, torch.Size]],
                 split_model: bool = False
                 ):
        super().__init__(encoder, embedding_size, hidden_dim, target_parameter, split_model)

        hidden_size = [hidden_dim, hidden_dim, hidden_dim]

        layers = []
        prev_size = embedding_size
        for h in hidden_size:
            layers.extend([
                nn.LeakyReLU(inplace=True),
                spectral_norm(nn.Linear(prev_size, h))
            ])
            prev_size = h
        layers.append(nn.LeakyReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)
        self.target_parameter = target_parameter

        self.task_head = nn.ModuleDict({
            k: ParameterGenerationBlock(self.hidden_dim, v) for k, v in target_parameter.items()
        })

    def forward(self, **kwargs) -> Dict:
        t = kwargs['task_name']
        del kwargs['task_name']

        if self.split_model:
            kwargs = {k: v.to('cuda:1') for k, v in kwargs.items()}
        x = self.encoder(**kwargs)[0][:, 0]

        if self.split_model:
            x = self.mlp(x)
            return self.task_head[t](x)
        else:
            x = self.mlp(x)
            return self.task_head[t](x)

    def to(self, *args, **kwargs):
        if self.split_model:
            self.encoder.to('cuda:1')
            self.mlp.to('cuda:1')
            self.task_head.to('cuda:1')
        else:
            super().to(*args, **kwargs)
