import logging
import os.path
import pickle
import time

import numpy as np
import pandas as pd
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import torch
from pandas import DataFrame
from sklearn import preprocessing
from torch import optim, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import Subset, TensorDataset
from ucimlrepo import fetch_ucirepo

from dataset.config import Datasets, dataset_config, get_transform
from model import get_pretrained_model
from model.mlp import MLP
from util.criteria import Compose, Accuracy, TopKAccuracy, CrossEntropy
from util.logger import logger
from util.trainer import seed_everything, get_dataloader, TrainingArguments, TrainerNoPred


def preprocess_dataset(d: Any) -> Tuple[TensorDataset, int, int]:
    x = d.data.features
    y = d.data.targets

    le_x = preprocessing.LabelEncoder()
    le_y = preprocessing.LabelEncoder()

    for col in x.columns:
        if x[col].dtype == object or isinstance(x[col].dtype, pd.CategoricalDtype):
            x[col] = x[col].fillna(x[col].mode()[0])
        else:
            x[col] = x[col].fillna(x[col].mean())

    for col in x.columns:
        if x[col].dtype == object or isinstance(x[col].dtype, pd.CategoricalDtype):
            x[col] = le_x.fit_transform(x[col])

    for col in y.columns:
        if y[col].dtype == object or isinstance(y[col].dtype, pd.CategoricalDtype) or (
            y[col].dtype == int and y[col].min() != 0):
            y[col] = le_y.fit_transform(y[col])

    x_norm = x.apply(lambda col: (col - col.min()) / (col.max() - col.min()))

    x_tensor = torch.from_numpy(x_norm.values).float()
    y_tensor = torch.from_numpy(y.values).squeeze()

    dataset = TensorDataset(x_tensor, y_tensor)

    return dataset, len(x.columns), y_tensor.unique().shape[0]


def main():
    parser = ArgumentParser()

    parser.add_argument('--output_dir', type=str, default=f'/root/data/model')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset', type=Datasets, choices=list(Datasets), default="53")

    parser.add_argument('--lr', '--learning_rate', type=float, default=2e-2)
    parser.add_argument('--wd', '--weight_decay', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, 'mlp', args.dataset.name)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{args.output_dir}/result.log', mode='w'))

    seed_everything(2024)

    d = fetch_ucirepo(id=int(args.dataset.value))

    dataset, in_dim, out_dim = preprocess_dataset(d)

    idx_perm = torch.randperm(len(dataset)).long()

    train_r, val_r, test_r = 0.6, 0.1, 0.3
    split = (idx_perm[:int(train_r * len(idx_perm))],
             idx_perm[int(train_r * len(idx_perm)):int(train_r * len(idx_perm)) + int(val_r * len(idx_perm))],
             idx_perm[int(train_r * len(idx_perm)) + int(val_r * len(idx_perm)):])

    with open(f'{args.output_dir}/split.bin', 'wb') as f:
        pickle.dump(split, f)

    train_dataset = Subset(dataset, split[0])
    eval_dataset = Subset(dataset, split[1])
    test_dataset = Subset(dataset, split[2])

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = get_dataloader(train_dataset, True, False, loader_kwargs)
    eval_loader = get_dataloader(eval_dataset, False, False, loader_kwargs)
    test_loader = get_dataloader(test_dataset, False, False, loader_kwargs)

    criteria = CrossEntropy()
    eval_metrics = Compose([CrossEntropy(), Accuracy()])

    def best_metric(cur: Dict, prev: Optional[Dict]):
        return prev is None or (cur['Acc']) > (prev['Acc'])

    model = MLP(in_dim=in_dim,
                out_dim=out_dim,
                hidden_dim=8,
                n_layers=4)

    train_args = TrainingArguments(do_train=True,
                                   do_eval=True,
                                   do_test=True,
                                   n_epochs=args.epoch,
                                   output_dir=args.output_dir,
                                   save_model=True)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ConstantLR(optimizer)
    trainer = TrainerNoPred(
        args=train_args,
        model=model,
        task_name=f'mlp_{args.dataset.name}',
        train_loader=train_loader,
        val_loader=eval_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        criteria=criteria,
        eval_metrics=eval_metrics,
        save_best=best_metric,
        preprocessing=lambda batch, device: ({'x': batch[0].to(device)}, batch[1].to(device))
    )
    trainer.exec()


if __name__ == '__main__':
    begin = time.time()
    main()
    end = time.time()
    logger.info(end - begin)
