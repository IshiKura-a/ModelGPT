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

from dataset.config import Datasets, dataset_config, get_transform
from model import get_pretrained_model
from model.hyper_network import LMMLPHyperNetwork
from model.summarizer import MockSummarizer, ImageCLSSummarizer
from util.criteria import CrossEntropy, Compose, Accuracy, TopKAccuracy
from util.logger import logger, print_args
from util.trainer import seed_everything, get_dataloader, collect_trainable_parameters, Trainer, TrainingArguments, \
    HyperTrainer


def main(rank: int, world_size: int, args: Namespace):
    # TODO: Support MP
    seed_everything(args.seed)

    def best_metric(cur: Dict[str, Dict], prev: Optional[Dict[str, Dict]]):
        keys = ['Acc', 'Acc@3', 'Acc@5']
        cur = np.mean(np.array([[v[k] for k in keys] for v in cur.values()]), axis=0)
        prev = np.mean(np.array([[v[k] for k in keys] for v in prev.values()]), axis=0) if prev else np.array([0, 0, 0])
        return tuple(cur) > tuple(prev)

    def preprocess_fn(batch: Any, device: str) -> Any:
        # Since we use MockSummarizer, we don't need to transform our images.
        # In case we directly communicate with GPT-4 during training, we need
        # to use `transform` to transform each image here.
        transform = transforms.Compose([
            transforms.ToPILImage(),
            ImageCLSSummarizer.img2base64
        ])
        return ({'x': normalize(batch[0].to(device), *d_conf.normalize)},
                batch[1].to(device),
                None)

    d_conf = dataset_config[args.dataset]
    task_list = tqdm(args.domain) if rank == 0 else args.domain
    target_parameter = {}
    trainers = []
    for idx, task_name in enumerate(task_list):
        criteria = CrossEntropy()
        eval_metrics = Compose([CrossEntropy(), Accuracy(), TopKAccuracy(3), TopKAccuracy(5)])

        with open(os.path.join(args.dataset_dir, task_name, 'split.bin'), 'rb') as f:
            split = pickle.load(f)
        d_dir = os.path.join(args.dataset_dir, task_name, 'images')

        d_train = ImageFolder(root=d_dir, transform=get_transform(d_conf.target_size, True, None))
        d_eval = ImageFolder(root=d_dir, transform=get_transform(d_conf.target_size, False, None))

        train_dataset = Subset(d_train, split[0])
        eval_dataset = Subset(d_eval, split[1])
        test_dataset = Subset(d_eval, split[2])

        loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}
        train_loader = get_dataloader(train_dataset, True, False, loader_kwargs)
        val_loader = {task_name: get_dataloader(eval_dataset, False, False, loader_kwargs)}
        test_loader = {task_name: get_dataloader(test_dataset, False, False, loader_kwargs)}

        model = get_pretrained_model(args.model, num_classes=d_conf.num_classes)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=args.target_modules,
            modules_to_save=args.modules_to_save,
        )
        model = get_peft_model(model, lora_config)

        def train(self, mode=True):
            type(model).train.__call__(self, mode)
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False
        model.train = functools.partial(train, model)

        p, t = collect_trainable_parameters(model)
        target_parameter[task_name] = {k: v.shape for k, v in t.items()}

        optimizer = optim.Adam(p, lr=args.t_lr, weight_decay=args.t_wd, amsgrad=True)
        scheduler = ConstantLR(optimizer)
        train_args = TrainingArguments(do_train=(task_name not in args.zero_shot_domain),
                                       do_eval=(task_name not in args.zero_shot_domain),
                                       do_test=True,
                                       n_epochs=0,
                                       rank=rank)
        trainer = Trainer(args=train_args,
                          model=model,
                          task_name=task_name,
                          train_loader=train_loader,
                          val_loader=val_loader,
                          test_loader=test_loader,
                          optimizer=optimizer,
                          scheduler=scheduler,
                          criteria=criteria,
                          eval_metrics=eval_metrics,
                          save_best=None,
                          n_local_updates=2 if task_name in ['dslr', 'webcam'] else 1,
                          preprocessing=preprocess_fn
                          )
        trainers.append(trainer)
        if rank == 0:
            trainable_params, all_param = model.get_nb_trainable_parameters()
            task_list.set_description(
                f'Current: {task_name} trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}')

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
    # net.load_state_dict(torch.load(f'/root/data/model/modelGPT/img_cls_zeroshot_AW/ResNet50Lora.pt'))
    logger.info(net)
    p, n = collect_trainable_parameters(net)
    optimizer = optim.Adam(p, lr=args.lr, weight_decay=args.wd, amsgrad=True)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    scheduler = ConstantLR(optimizer)
    h_trainer = HyperTrainer(args=train_args,
                             model=net,
                             summarizer=MockSummarizer(f'./blob/img_cls.json'),
                             target_trainer=trainers,
                             task_name='ResNet50Lora',
                             tokenizer=tokenizer,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             preprocessing=preprocess_fn,
                             n_local_updates=args.n_local_updates,
                             save_best=best_metric,
                             exec_finetune=args.exec_finetune,
                             finetune_epoch=args.finetune_epoch
                             )
    h_trainer.exec()


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    parser = ArgumentParser()

    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--backbone', type=str, default='/root/data/model/distilbert-base-uncased')
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default=f'/root/data/model/modelGPT/img_cls_zeroshot_AD')

    parser.add_argument('--model', type=str, choices=list_models(), default='resnet50')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--wd', type=float, default=1e-5)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--t_lr', type=float, default=1e-3)
    parser.add_argument('--t_wd', type=float, default=1e-3)
    parser.add_argument('--n_local_updates', type=int, default=1)

    parser.add_argument('--dataset', type=Datasets, choices=list(Datasets), default=Datasets.Office)
    parser.add_argument('--domain', type=str, nargs='+', default=['amazon', 'dslr', 'webcam'])
    parser.add_argument('--zero_shot_domain', type=str, nargs='+', default=['webcam'])
    parser.add_argument('--dataset_dir', type=str, default=f'/root/data/dataset/office-31/')

    parser.add_argument('--lora_r', type=int, default=8)
    parser.add_argument('--lora_alpha', type=int, default=16)
    parser.add_argument('--lora_dropout', type=int, default=0.1)
    parser.add_argument('--target_modules', type=str, default=r'layer.\..\.conv.')
    parser.add_argument('--modules_to_save', type=str, default=r'fc')

    parser.add_argument('--exec_finetune', action='store_true', default=True)
    parser.add_argument('--finetune_epoch', type=int, default=1)

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    # logger.addHandler(logging.FileHandler(f'{args.output_dir}/result.log', mode='w'))
    print_args(args)
    main(0, 1, args)
