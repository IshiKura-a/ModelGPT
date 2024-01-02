from enum import Enum
from typing import List, Optional

from torchvision import transforms

DEFAULT_NORMALIZE = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]


class DatasetConfig:
    def __init__(self, target_size: int, num_classes: int, normalize: Optional[List]):
        self.target_size = target_size
        self.num_classes = num_classes
        self.normalize = normalize


class Datasets(Enum):
    VisDA2017 = 'visda2017'
    Office = 'office'
    OfficeHome = 'office_home'
    CIFAR100 = 'cifar100'
    CIFAR10 = 'cifar10'
    Caltech256 = 'caltech256'
    Iris = '53'
    HeartDisease = '45'
    Wine = '109'
    Adult = '2'
    BreastCancer = '17'
    CarEvaluation = '19'
    WineQuality = '186'
    DryBean = '602'
    Rice = '545'
    BankMarketing = '222'
    Default = 'default'


def get_transform(img_size: int, is_train: bool, normalize: Optional[List] = DEFAULT_NORMALIZE) -> transforms.Compose:
    transform = [
        transforms.Resize((img_size, img_size)),
        transforms.RandomCrop((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ] if is_train else [
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor()
    ]
    if normalize is not None:
        transform.append(transforms.Normalize(*normalize))
    return transforms.Compose(transform)


dataset_config = {
    Datasets.VisDA2017: DatasetConfig(
        target_size=32,
        num_classes=12,
        normalize=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    ),
    Datasets.Office: DatasetConfig(
        target_size=32,
        num_classes=31,
        normalize=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    ),
    Datasets.OfficeHome: DatasetConfig(
        target_size=32,
        num_classes=65,
        normalize=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    ),
    Datasets.CIFAR100: DatasetConfig(
        target_size=32,
        num_classes=100,
        normalize=[[0.5071, 0.4867, 0.4408], [0.2675, 0.2565, 0.2761]]
    ),
    Datasets.Caltech256: DatasetConfig(
        target_size=32,
        num_classes=257,
        normalize=[[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
    ),
    Datasets.CIFAR10: DatasetConfig(
        target_size=32,
        num_classes=10,
        normalize=[[0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]]
    )
}
