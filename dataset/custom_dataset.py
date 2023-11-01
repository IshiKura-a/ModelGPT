import os
from typing import List, Tuple, Any, Union, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CustomBatch:
    def __init__(self, data: Any):
        transposed_data = list(zip(*data))
        self.data = [torch.stack(d, 0) for d in transposed_data]

    def pin_memory(self) -> 'CustomBatch':
        self.data = [
            d.pin_memory() for d in self.data
        ]
        return self


def collate_wrapper(batch: Any) -> CustomBatch:
    return CustomBatch(batch)


class CompositeDataset(Dataset):
    def __init__(self, datasets: List[Dataset]):
        assert len(list(set([len(d) for d in datasets]))) == 1, "Datasets should be the same length"
        self.datasets = datasets

    def __getitem__(self, item):
        x = []
        for d in self.datasets:
            i = d.__getitem__(item)
            if isinstance(i, tuple):
                x += list(i)
            else:
                x.append(i)
        return tuple(x)

    def __len__(self):
        return len(self.datasets[0])


class CustomDataset(Dataset):
    def __init__(self, root_dir: str, transform: transforms.Compose):
        self.root_dir = root_dir
        self.transform = transform

        attr_df = pd.read_csv(os.path.join(root_dir, 'image_list.txt'),
                              header=None, delimiter=' ', names=['filename', 'y'])
        self.filename_arr = attr_df.loc[:, 'filename']
        self.y_arr = attr_df.loc[:, 'y']

    def __len__(self):
        return len(self.filename_arr)

    def __getitem__(self, item):
        y = self.y_arr[item]
        img_name = os.path.join(self.root_dir, self.filename_arr[item])
        img = Image.open(img_name).convert('RGB')
        return self.transform(img), y
