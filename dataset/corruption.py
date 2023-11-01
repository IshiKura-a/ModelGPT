from enum import Enum
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset


class Corruption(Enum):
    Brightness = "brightness"
    DefocusBlur = "defocus_blur"
    Frost = "frost"
    GlassBlur = "glass_blur"
    Saturate = "saturate"
    Spatter = "spatter"
    ElasticTransform = "elastic_transform"
    GaussianBlur = "gaussian_blur"
    ImpulseNoise = "impulse_noise"
    MotionBlur = "motion_blur"
    ShotNoise = "shot_noise"
    SpeckleNoise = "speckle_noise"
    Contrast = "contrast"
    Fog = "fog"
    GaussianNoise = "gaussian_noise"
    JpegCompression = "jpeg_compression"
    Pixelate = "pixelate"
    Snow = "snow"
    ZoomBlur = "zoom_blur"


def get_corruption_dataset(root_dir: str, corruption: Corruption, severity: int) -> TensorDataset:
    assert 1 <= severity <= 5
    n_cifar = 10000
    label_path = Path(root_dir) / 'labels.npy'
    img_path = Path(root_dir) / (corruption.value + '.npy')

    labels = torch.tensor(np.load(label_path)[severity * n_cifar: (severity + 1) * n_cifar])
    imgs = np.load(img_path)[severity * n_cifar: (severity + 1) * n_cifar]
    imgs = torch.tensor(np.transpose(imgs, (0, 3, 1, 2)).astype(np.float32) / 255)

    return TensorDataset(imgs, labels)


d = get_corruption_dataset('/data/home/tangzihao/dataset/cifar-10-c/', Corruption.Snow, 1)
print(d[0])
print(d.__len__())
