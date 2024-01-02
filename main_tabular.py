import functools
import logging
import os
import pickle
from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch import optim, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import list_models, ResNet
from torchvision.transforms.functional import normalize
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from ucimlrepo import fetch_ucirepo

from baseline.tabular_cls import preprocess_dataset
from dataset.config import Datasets, dataset_config, get_transform
from model import get_pretrained_model
from model.hyper_network import LMMLPHyperNetwork
from model.mlp import MLP
from model.summarizer import MockSummarizer, ImageCLSSummarizer
from util.criteria import CrossEntropy, Compose, Accuracy, TopKAccuracy
from util.logger import logger, print_args
from util.trainer import seed_everything, get_dataloader, collect_trainable_parameters, Trainer, TrainingArguments, \
    HyperTrainer


def get_n_local_updates(d: Datasets):
    if d in [Datasets.Wine, Datasets.Iris]:
        return 10
    elif d in [Datasets.HeartDisease, Datasets.Rice, Datasets.DryBean]:
        return 2
    else:
        return 1


def main(rank: int, world_size: int, args: Namespace):
    # TODO: Support MP
    seed_everything(args.seed)

    def best_metric(cur: Dict[str, Dict], prev: Optional[Dict[str, Dict]]):
        keys = ['Acc']
        cur = np.mean(np.array([[v[k] for k in keys] for v in cur.values()]), axis=0)
        prev = np.mean(np.array([[v[k] for k in keys] for v in prev.values()]), axis=0) if prev else np.array([0, 0, 0])
        return tuple(cur) > tuple(prev)

    def preprocess_fn(batch: Any, device: str) -> Any:
        # Since we use MockSummarizer, we don't need to transform
        return ({'x': batch[0].to(device)},
                batch[1].to(device),
                None)

    task_list = tqdm(args.dataset) if rank == 0 else args.dataset
    target_parameter = {}
    trainers = []
    for idx, task in enumerate(task_list):
        criteria = CrossEntropy()
        eval_metrics = Compose([CrossEntropy(), Accuracy()])

        with open(os.path.join(args.dataset_dir, task.name, 'split.bin'), 'rb') as f:
            split = pickle.load(f)

        d = fetch_ucirepo(id=int(task.value))
        dataset, in_dim, out_dim = preprocess_dataset(d)

        train_dataset = Subset(dataset, split[0])
        eval_dataset = Subset(dataset, split[1])
        test_dataset = Subset(dataset, split[2])

        loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
        train_loader = get_dataloader(train_dataset, True, False, loader_kwargs)
        val_loader = {task.name: get_dataloader(eval_dataset, False, False, loader_kwargs)}
        test_loader = {task.name: get_dataloader(test_dataset, False, False, loader_kwargs)}

        model = MLP(in_dim=in_dim,
                    out_dim=out_dim,
                    hidden_dim=args.t_hidden_dim,
                    n_layers=args.t_n_layers)

        p, t = collect_trainable_parameters(model)
        target_parameter[task.name] = {k: v.shape for k, v in t.items()}

        optimizer = optim.Adam(p, lr=args.t_lr, weight_decay=args.t_wd, amsgrad=True)
        scheduler = ConstantLR(optimizer)
        train_args = TrainingArguments(do_train=True,
                                       do_eval=True,
                                       do_test=True,
                                       n_epochs=0,
                                       rank=rank)
        trainer = Trainer(args=train_args,
                          model=model,
                          task_name=task.name,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          criteria=criteria,
                          eval_metrics=eval_metrics,
                          save_best=None,
                          n_local_updates=get_n_local_updates(task),
                          preprocessing=preprocess_fn
                          )
        trainers.append(trainer)
        if rank == 0:
            task_list.set_description(f'Current: {task.name}')

    train_args = TrainingArguments(do_train=False,
                                   do_eval=False,
                                   do_test=True,
                                   output_dir=args.output_dir,
                                   n_epochs=args.epoch,
                                   save_model=True,
                                   rank=rank)
    encoder = AutoModel.from_pretrained(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    if rank == 0:
        logger.info(f'target_parameters: {target_parameter[list(target_parameter.keys())[0]].keys()}')

    net = LMMLPHyperNetwork(encoder=encoder,
                            embedding_size=768,
                            hidden_dim=args.hidden_dim,
                            target_parameter=target_parameter
                            )
    net.encoder.requires_grad_(False)
    logger.info(net)
    p, n = collect_trainable_parameters(net)
    optimizer = optim.Adam(p, lr=args.lr, weight_decay=args.wd, amsgrad=True)
    scheduler = ConstantLR(optimizer)
    h_trainer = HyperTrainer(args=train_args,
                             model=net,
                             summarizer=MockSummarizer(f'./blob/tabular.json'),
                             target_trainer=trainers,
                             task_name='MLP',
                             tokenizer=tokenizer,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             preprocessing=preprocess_fn,
                             n_local_updates=args.n_local_updates,
                             save_best=best_metric,
                             exec_finetune=True
                             )
    h_trainer.exec()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--backbone', type=str, default='/root/data/model/distilbert-base-uncased')
    parser.add_argument('--hidden_dim', type=int, default=25)
    parser.add_argument('--output_dir', type=str, default=f'/root/data/model/modelGPT/tabular_hyn')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=80)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--t_lr', type=float, default=2e-2)
    parser.add_argument('--t_wd', type=float, default=1e-4)
    parser.add_argument('--t_hidden_dim', type=int, default=8)
    parser.add_argument('--t_n_layers', type=int, default=4)
    parser.add_argument('--n_local_updates', type=int, default=1)
    parser.add_argument('--dataset', type=Datasets, nargs='+', choices=list(Datasets),
                        default=[Datasets.Iris, Datasets.HeartDisease, Datasets.Wine, Datasets.Adult,
                                 Datasets.BreastCancer, Datasets.CarEvaluation, Datasets.WineQuality, Datasets.DryBean,
                                 Datasets.Rice, Datasets.BankMarketing])
    parser.add_argument('--dataset_dir', type=str, default='/root/data/model/mlp')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # logger.addHandler(logging.FileHandler(f'{args.output_dir}/result.log', mode='w'))
    print_args(args)
    main(0, 1, args)
