import argparse
import logging
import os
import tracemalloc
from pathlib import Path
from typing import Any, Tuple, Dict, Optional

import torch
from datasets import load_dataset, Dataset, DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim, is_tensor, nn
from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR
from torch.utils.data import DataLoader, DistributedSampler
from torch import multiprocessing as mp
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from torchsummary import summary
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedTokenizer, AutoModel
# import bitsandbytes as bnb
from baseline.glue import preprocess_fn
from dataset.glue import get_preprocess_fn, get_num_labels
from model.hyper_network import DistilBertFCHYN, HyperNetwork, MultiHeadLMFCHYN, LMMLPHyperNetwork
from model.summarizer import MockSummarizer
from util.criteria import CrossEntropy, Accuracy, Compose, MSELoss
from util.logger import print_args, logger
from util.trainer import TrainingArguments, Trainer, seed_everything, HyperTrainer, get_dataloader, \
    collect_parameters, collect_trainable_parameters


def get_n_local_updates(task_name: str):
    if task_name in ['wnli', 'rte']:
        return 5
    elif task_name in ['stsb', 'cola', 'mrpc']:
        return 3
    else:
        return 1


def task_init(task_name: str, args: argparse.Namespace) -> Tuple[nn.Module, PreTrainedTokenizer, DatasetDict]:
    d = load_dataset('/root/data/dataset/glue', task_name)
    num_labels = get_num_labels(task_name)

    config = AutoConfig.from_pretrained(
        args.backbone,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, config=config)
    d = d.map(
        get_preprocess_fn(task_name, tokenizer, 128),
        batched=True,
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )
    d.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)

    return model, tokenizer, d


def main(rank: int, world_size: int, args: argparse.Namespace):
    tracemalloc.start()
    if args.mp:
        ddp_setup(rank, world_size)
    seed_everything(args.seed)

    loader_kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'pin_memory': True}

    trainers = []
    target_parameter = {}
    # task_list = ['stsb']
    task_list = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']
    # task_list = ['cola', 'mrpc',  'rte', 'stsb', 'wnli']
    if rank == 0:
        task_list = tqdm(task_list)
    for idx, task_name in enumerate(task_list):
        num_labels = get_num_labels(task_name)
        if num_labels == 1:
            criteria = MSELoss()
            eval_metrics = Compose([MSELoss()])
        else:
            criteria = CrossEntropy()
            eval_metrics = Compose([CrossEntropy(), Accuracy()])
        model, tokenizer, d = task_init(task_name, args)
        val_key = 'validation_matched' if task_name == 'mnli' else 'validation'
        test_key = 'test_matched' if task_name == 'mnli' else 'test'
        train_loader = get_dataloader(d['train'], True, args.mp, loader_kwargs)
        val_loader = {task_name: get_dataloader(d[val_key], False, args.mp, loader_kwargs)}
        test_loader = {task_name: get_dataloader(d[test_key], False, args.mp, loader_kwargs)}

        if task_name == 'mnli':
            _, _, ax_d = task_init('ax', args)
            val_loader['mnli-mm'] = get_dataloader(d['validation_mismatched'], False, args.mp, loader_kwargs)
            test_loader['mnli-mm'] = get_dataloader(d['test_mismatched'], False, args.mp, loader_kwargs)
            test_loader['ax'] = get_dataloader(ax_d['test'], False, args.mp, loader_kwargs)
        train_args = TrainingArguments(do_train=True,
                                       do_eval=True,
                                       do_test=True,
                                       n_epochs=0,
                                       mp=args.mp,
                                       rank=rank)

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
            target_modules=args.target_modules
        )
        model = get_peft_model(model, lora_config)
        p, t = collect_trainable_parameters(model)
        target_parameter[task_name] = {k: v.shape for k, v in t.items()}
        # model = DDP(model.cuda()) if args.mp else model.cuda()
        optimizer = optim.Adam(p, lr=args.t_lr, weight_decay=args.t_wd, amsgrad=True)
        scheduler = ConstantLR(optimizer)
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
                          n_local_updates=get_n_local_updates(task_name),
                          preprocessing=preprocess_fn
                          )
        trainers.append(trainer)

        if rank == 0:
            trainable_params, all_param = model.get_nb_trainable_parameters()
            task_list.set_description(
                f'Current: {task_name} trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}')
        del tokenizer

    train_args = TrainingArguments(do_train=True,
                                   do_eval=True,
                                   do_test=True,
                                   output_dir=args.output_dir,
                                   n_epochs=args.epoch,
                                   save_model=True,
                                   mp=args.mp,
                                   rank=rank,
                                   is_pred=True)
    encoder = AutoModel.from_pretrained(args.backbone)
    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    if rank == 0:
        logger.info(f'target_parameters: {target_parameter[list(target_parameter.keys())[0]].keys()}')

    def best_metric(cur: Dict, prev: Optional[Dict]):
        cal_avg_loss = lambda x: sum([v['Acc'] if 'Acc' in v else -v['MSE'] for v in x.values()]) / len(x)
        if prev is None:
            return True
        else:
            return cal_avg_loss(cur) > cal_avg_loss(prev)

    net = LMMLPHyperNetwork(encoder=encoder,
                            embedding_size=768,
                            hidden_dim=args.hidden_dim,
                            target_parameter=target_parameter,
                            )
    net.load_state_dict(torch.load(f'/root/data/model/modelGPT/distilbert_lora_hyn_mh/DistilBertLora.pt'))
    logger.info(net)
    net = DDP(net.cuda(), gradient_as_bucket_view=True) if args.mp else net.cuda()
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd, amsgrad=True)
    # p, _ = collect_parameters(net, ['mlp', 'task_head'])
    # optimizer = optim.Adam(p, lr=args.lr, weight_decay=args.wd)
    # scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch)
    scheduler = ConstantLR(optimizer)
    h_trainer = HyperTrainer(args=train_args,
                             model=net,
                             summarizer=MockSummarizer(f'./blob/seq_cls.json'),
                             target_trainer=trainers,
                             task_name='DistilBertLora',
                             tokenizer=tokenizer,
                             optimizer=optimizer,
                             scheduler=scheduler,
                             preprocessing=preprocess_fn,
                             n_local_updates=args.n_local_updates,
                             save_best=best_metric,
                             exec_finetune=True
                             )
    h_trainer.exec()
    if args.mp:
        dist.destroy_process_group()


def ddp_setup(rank: int, world_size: int):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '19635'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    parser = argparse.ArgumentParser()

    parser.add_argument('--mp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--backbone', type=str, default='/root/data/model/distilbert-base-uncased')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--output_dir', type=str, default=f'/root/data/model/modelGPT/distilbert_lora_hyn_0118')

    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--wd', type=float, default=0.0001)
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--t_lr', type=float, default=1e-4)
    parser.add_argument('--t_wd', type=float, default=1e-4)
    parser.add_argument('--n_local_updates', type=int, default=1)

    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--target_modules', type=str, default=r'.*[qv]_lin')

    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.addHandler(logging.FileHandler(f'{args.output_dir}/result.log', mode='w'))
    print_args(args)
    if args.mp:
        world_size = torch.cuda.device_count()
        mp.spawn(main, (world_size, args), nprocs=world_size)
    else:
        main(0, 1, args)
