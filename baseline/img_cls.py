import functools
import logging
import os.path
import pickle
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional, Dict

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision.models import list_models
from torchvision.transforms.functional import normalize

from dataset.config import Datasets, dataset_config, get_transform
from model import get_pretrained_model
from util.criteria import Compose, Accuracy, TopKAccuracy, CrossEntropy
from util.logger import logger
from util.trainer import seed_everything, get_dataloader, TrainingArguments, TrainerNoPred


def main():
    parser = ArgumentParser()

    parser.add_argument('--output_dir', type=str, default=f'/root/data/model')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--dataset', type=Datasets, choices=list(Datasets), default=Datasets.Office)
    parser.add_argument('--domain', type=str, default='webcam')
    parser.add_argument('--dataset_dir', type=str, default=f'/root/data/dataset/office-31/webcam/images')

    parser.add_argument('--model', type=str, choices=list_models(), default='resnet50')
    parser.add_argument('--lr', '--learning_rate', type=float, default=1e-3)
    parser.add_argument('--wd', '--weight_decay', type=float, default=0)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)

    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=int, default=0.1)
    parser.add_argument('--target_modules', type=str, default=r'layer.\..\.conv.')
    parser.add_argument('--modules_to_save', type=str, default=r'fc')
    parser.add_argument('--load_ckpt', action='store_true', default=False)

    args = parser.parse_args()
    args.output_dir = os.path.join(args.output_dir, args.model, args.dataset.value, args.domain, str(args.seed))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{args.output_dir}/result.log', mode='w'))

    seed_everything(args.seed)
    config = dataset_config[args.dataset]

    d_train = ImageFolder(root=args.dataset_dir, transform=get_transform(config.target_size, True, None))
    d_eval = ImageFolder(root=args.dataset_dir, transform=get_transform(config.target_size, False, None))

    idx_perm = torch.randperm(len(d_train)).long()

    split = (idx_perm[:int(0.8 * len(idx_perm))],
             idx_perm[int(0.8 * len(idx_perm)):int(0.8 * len(idx_perm)) + int(0.1 * len(idx_perm))],
             idx_perm[int(0.8 * len(idx_perm)) + int(0.1 * len(idx_perm)):])

    with open(f'{args.output_dir}/split.bin', 'wb') as f:
        pickle.dump(split, f)

    train_dataset = Subset(d_train, split[0])
    eval_dataset = Subset(d_eval, split[1])
    test_dataset = Subset(d_eval, split[2])

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
    train_loader = get_dataloader(train_dataset, True, False, loader_kwargs)
    eval_loader = get_dataloader(eval_dataset, False, False, loader_kwargs)
    test_loader = get_dataloader(test_dataset, False, False, loader_kwargs)

    criteria = CrossEntropy()
    eval_metrics = Compose([CrossEntropy(), Accuracy(), TopKAccuracy(3), TopKAccuracy(5)])

    def best_metric(cur: Dict, prev: Optional[Dict]):
        return prev is None or (cur['Acc'], cur['Acc@3'], cur['Acc@5']) > (prev['Acc'], prev['Acc@3'], prev['Acc@5'])

    model = get_pretrained_model(args.model, num_classes=config.num_classes)
    if args.load_ckpt:
        model.load_state_dict(torch.load(f'/root/data/model/modelGPT/img_cls_zeroshot_128/webcam.pt'))

        def train(self, mode=True):
            type(model).train.__call__(self, mode)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        model.train = functools.partial(train, model)
    train_args = TrainingArguments(do_train=True,
                                   do_eval=True,
                                   do_test=True,
                                   n_epochs=args.epoch,
                                   output_dir=args.output_dir,
                                   save_model=True)
    # lora_config = LoraConfig(
    #     r=args.lora_r,
    #     lora_alpha=args.lora_alpha,
    #     lora_dropout=args.lora_dropout,
    #     bias="none",
    #     target_modules=args.target_modules,
    #     modules_to_save=args.modules_to_save
    # )
    # model = get_peft_model(model, lora_config)
    # trainable_params, all_param = model.get_nb_trainable_parameters()
    # logger.info(
    #     f'trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}')
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = ConstantLR(optimizer)
    trainer = TrainerNoPred(
        args=train_args,
        model=model,
        task_name=f'{args.model}_{args.dataset.value}_{args.domain}',
        train_loader=train_loader,
        val_loader=eval_loader,
        test_loader=test_loader,
        scheduler=scheduler,
        optimizer=optimizer,
        criteria=criteria,
        eval_metrics=eval_metrics,
        save_best=best_metric,
        preprocessing=lambda batch, device: (
            {'x': normalize(batch[0].to(device), *config.normalize)}, batch[1].to(device))
    )
    trainer.exec()


if __name__ == '__main__':
    begin = time.time()
    main()
    end = time.time()
    logger.info(end-begin)
