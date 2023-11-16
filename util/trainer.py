import collections
import os
import random
import time
from abc import abstractmethod, ABC
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from pickle import dump
from typing import Optional, Any, Union, Tuple, List, Dict, Callable

import numpy as np
import torch
from datasets import Dataset
from torch import nn, Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from dataset.glue import get_num_labels
from model.summarizer import Summarizer
from util import rsetattr
from util.criteria import Metric, Compose
from util.logger import logger


class TrainingArguments:
    output_dir: Optional[str]
    save_model: bool
    n_epochs: int = 0
    do_train: bool
    do_eval: bool
    do_test: bool
    evaluation_every: int = 1
    resume_from_ckpt: Optional[str]
    show_progress: bool
    device: str
    mp: bool
    rank: int
    deepspeed: bool
    fp16: bool

    def __init__(self,
                 do_train: bool,
                 do_eval: bool,
                 do_test: bool,
                 output_dir: Optional[str] = None,
                 save_model: bool = False,
                 n_epochs: int = 0,
                 evaluation_every: int = 1,
                 resume_from_ckpt: Optional[str] = None,
                 show_progress: bool = True,
                 device: str = 'cuda',
                 mp: bool = False,
                 rank: int = 0,
                 use_deepspeed: bool = False,
                 fp16: bool = False
                 ):
        self.do_train = do_train
        self.do_eval = do_eval
        self.do_test = do_test
        self.output_dir = output_dir
        self.save_model = save_model and (rank == 0)
        self.n_epochs = n_epochs
        self.evaluation_every = evaluation_every
        self.resume_from_ckpt = resume_from_ckpt
        self.show_progress = show_progress and (rank == 0)
        self.device = device
        self.mp = mp
        self.rank = rank
        self.deepspeed = use_deepspeed
        self.fp16 = fp16


class BaseTrainer(ABC):
    @staticmethod
    def default_fn(batch: Any, device: str) -> Any:
        return tuple(t.to(device) for t in batch)

    model: nn.Module
    train_loader: Any
    val_loader: Any
    test_loader: Any

    scheduler: Any
    optimizer: Optional[Optimizer]
    criteria: Optional[Metric]
    eval_metrics: Optional[Metric]
    save_best: Optional[Callable[[Any, Any], bool]]
    args: Optional[TrainingArguments]
    preprocessing: Callable[[Any, str], Any]

    def __init__(self, model: nn.Module,
                 task_name: str,
                 args: Optional[TrainingArguments],
                 train_loader: Any,
                 val_loader: Any,
                 test_loader: Any,
                 scheduler: Any,
                 optimizer: Optional[Optimizer],
                 criteria: Optional[Metric],
                 eval_metrics: Optional[Metric],
                 save_best: Optional[Callable[[Any, Any], bool]],
                 preprocessing: Callable[[Any, str], Any] = default_fn
                 ):
        if not (train_loader or not args.do_train) and \
            (val_loader or not args.do_eval) and \
            (test_loader or not args.do_test):
            logger.warning(f'Possible no data is provided, if not convinced, plz check the trainer created.')

        self.model = model
        self.task_name = task_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.scheduler = scheduler
        self.optimizer = optimizer

        self.criteria = criteria
        self.eval_metrics = eval_metrics
        self.save_best = save_best
        self.args = args
        self.preprocessing = lambda x: preprocessing(x, self.args.device)

        self.model.to(args.device)
        if args.output_dir is not None:
            Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    def set_train(self, models: List[Optional[nn.Module]]):
        for model in models:
            if model is not None:
                model.to(self.args.device)
                model.train()

    def set_eval(self, models: List[Optional[nn.Module]]):
        for model in models:
            if model is not None:
                model.to(self.args.device)
                model.eval()

    @abstractmethod
    def train(self, **kwargs) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def eval(self, **kwargs) -> Union[Metric, Tuple[Metric, Tensor]]:
        raise NotImplementedError

    @abstractmethod
    def pred(self, **kwargs) -> Tensor:
        raise NotImplementedError

    @abstractmethod
    def exec(self, **kwargs) -> Dict:
        raise NotImplementedError

    @property
    def get_model(self) -> nn.Module:
        return self.model

    def _save_checkpoint(self, res: Dict):
        if self.args.rank != 0:
            return
        torch.save(res, os.path.join(self.args.output_dir, f'{self.task_name}_ckpt.pt'))
        if self.args.mp:
            torch.save(self.model.module.state_dict(), os.path.join(self.args.output_dir, f'{self.task_name}.pt'))
        else:
            torch.save(self.model.state_dict(), os.path.join(self.args.output_dir, f'{self.task_name}.pt'))

    def _load_checkpoint(self):
        if self.args.deepspeed:
            _, res = self.model.load_checkpoint(self.args.output_dir, f'{self.task_name}_ds')
            return res
        else:
            res = torch.load(os.path.join(self.args.output_dir, f'{self.task_name}_ckpt.pt'))
            self.model.load_state_dict(torch.load(res['state_dict_dir'], map_location=lambda storage, loc: storage))
            return res


class GLUETrainer(BaseTrainer):
    def train(self, epoch: int) -> Dict:
        self.set_train([self.model])

        loader = tqdm(self.train_loader, total=len(self.train_loader)) if self.args.show_progress else self.train_loader

        self.criteria.reset()
        if self.eval_metrics is not None:
            self.eval_metrics.reset()

        tot_loss = 0
        with torch.set_grad_enabled(True):
            for batch_idx, batch in enumerate(loader):
                batch = self.preprocessing(batch)
                x = batch[0]
                y = batch[1]

                outputs = self.model(**x).logits
                loss = self.criteria.update(outputs, y).get_results()
                if self.eval_metrics is not None:
                    with torch.no_grad():
                        self.eval_metrics.update(outputs, y)

                tot_loss += loss.detach().cpu().numpy()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.criteria.reset()

                del batch
                if self.args.show_progress:
                    loader.set_description('epoch: {:04} loss = {:8.3f} {}'.format(epoch, loss, self.eval_metrics))

        if isinstance(self.eval_metrics, Compose):
            metric_dict = self.eval_metrics.get_results()
        else:
            metric_dict = {self.eval_metrics.name: self.eval_metrics.get_results()}
        return {
            'epoch': epoch,
            'loss': tot_loss,
            **metric_dict
        } if self.eval_metrics is not None else {
            'epoch': epoch,
            'loss': tot_loss
        }

    def eval(self, epoch: int,
             loader: DataLoader,
             return_results: bool = False) -> Union[Metric, Tuple[Metric, Tensor]]:
        self.set_eval([self.model])

        loader = tqdm(loader, total=len(loader)) if self.args.show_progress else loader

        self.eval_metrics.reset()
        results = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = self.preprocessing(batch)
                x = batch[0]
                y = batch[1]

                outputs = self.model(**x).logits
                self.eval_metrics.update(outputs, y)
                results.append(outputs)

                if self.args.show_progress:
                    loader.set_description('[Eval] epoch: {:04} {}'.format(epoch, self.eval_metrics))

        results = torch.cat(results).to(self.args.device)
        if results.shape[-1] > 1:
            results = results.softmax(dim=-1)
        return (self.eval_metrics, results.squeeze()) if return_results else self.eval_metrics

    def pred(self, loader: DataLoader) -> Tensor:
        self.set_eval([self.model])

        loader = tqdm(loader, total=len(loader)) if self.args.show_progress else loader

        results = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                batch = self.preprocessing(batch)
                x = batch[0]

                outputs = self.model(**x).logits
                results.append(outputs)

        results = torch.cat(results).to(self.args.device)
        if results.shape[-1] > 1:
            results = results.softmax(dim=-1).argmax(dim=-1)
        return results.squeeze()

    def exec(self) -> Dict:
        epoch_offset = 0
        if self.args.resume_from_ckpt is not None:
            raise NotImplementedError

        train_results = []
        best_result = None
        res = {}
        for epoch in range(epoch_offset, self.args.n_epochs):
            if self.args.do_train:
                results = self.train(epoch)
                train_results.append(results)

            if self.args.do_eval and epoch % self.args.evaluation_every == 0:
                eval_metrics = self.eval(epoch, self.val_loader, False)

            if self.args.save_model:
                result = eval_metrics.get_results() if self.args.do_eval else train_results[-1]['loss']
                if self.save_best is None or self.save_best(result, best_result):
                    logger.info(f'Best is saved at epoch {epoch}')
                    best_result = result
                    res = {
                        'epoch': epoch,
                        'optim': self.optimizer.state_dict(),
                        'sched': self.scheduler.state_dict(),
                        'metrics': eval_metrics.get_results() if eval_metrics is not None else '',
                        'state_dict_dir': os.path.join(self.args.output_dir, f'{self.task_name}.pt'),
                    }
                    self._save_checkpoint(res)

        res['train'] = train_results
        if self.args.save_model and epoch_offset < self.args.n_epochs:
            self.model.load_state_dict(
                torch.load(os.path.join(self.args.output_dir, f'{self.task_name}.pt')))

        if self.args.do_eval:
            _ = self.eval(-1, self.val_loader, False)
        if self.args.do_test:
            results = self.pred(self.test_loader)
            results = results.clone().detach().cpu().numpy()

            with open(os.path.join(self.args.output_dir, f'{self.task_name}.tsv'), 'w') as f:
                print(f'index\tprediction', file=f)
                for i in range(results.shape[0]):
                    print(f'{i}\t{results[i]}', file=f)

        if self.args.save_model:
            try:
                ckpt = torch.load(os.path.join(self.args.output_dir, f'{self.task_name}_ckpt.pt'))
                ckpt['epoch'] = self.args.n_epochs
                torch.save(ckpt, os.path.join(self.args.output_dir, f'{self.task_name}_ckpt.pt'))
            except FileNotFoundError:
                pass
        return res


class HyperTrainer(BaseTrainer):
    def __init__(self, model: nn.Module,
                 task_name: str,
                 summarizer: Summarizer,
                 tokenizer: PreTrainedTokenizer,
                 target_trainer: List[BaseTrainer],
                 args: Optional[TrainingArguments],
                 n_local_updates: int = 3,
                 train_loader: Optional[List[DataLoader]] = None,
                 val_loader: Optional[List[DataLoader]] = None,
                 test_loader: Optional[List[DataLoader]] = None,
                 scheduler: Any = None,
                 optimizer: Optional[Optimizer] = None,
                 criteria: Optional[Metric] = None,
                 eval_metrics: Optional[Metric] = None,
                 save_best: Optional[Callable[[Any, Any], bool]] = None,
                 preprocessing: Callable[[Any, str], Any] = BaseTrainer.default_fn):
        super().__init__(model, task_name, args, train_loader, val_loader, test_loader, scheduler, optimizer, criteria,
                         eval_metrics, save_best, preprocessing)

        self.target_trainer = target_trainer
        self.summarizer = summarizer
        self.tokenizer = tokenizer
        self.n_local_updates = n_local_updates

    def train(self, task: int, epoch: int):
        tt = self.target_trainer[task]
        loader = tt.train_loader
        self.set_train([self.model, tt.model])

        if self.args.mp:
            loader.sampler.set_epoch(epoch)

        loader = tqdm(loader, total=len(loader)) if self.args.show_progress else loader

        with torch.set_grad_enabled(True):
            for batch_idx, batch in enumerate(loader):
                batch = self.preprocessing(batch)
                x = batch[0]
                y = batch[1]
                z = batch[2]

                pattern = self.summarizer(z, task_name=[tt.task_name])
                pattern = self.tokenizer(pattern, padding='max_length', max_length=128, truncation=True,
                                         return_tensors='pt')
                pattern = {k: v.to(self.args.device) for k, v in pattern.items()}

                outputs = self.model(**pattern, task_name=tt.task_name)
                d = tt.model.state_dict()
                d.update(outputs)
                tt.model.load_state_dict(d)

                tt.eval_metrics.reset()
                for _ in range(self.n_local_updates):
                    tt.criteria.reset()
                    t_outputs = tt.model(**x).logits
                    local_loss = tt.criteria.update(t_outputs, y).get_results()
                    with torch.no_grad():
                        tt.eval_metrics.update(t_outputs, y)
                        if self.args.show_progress:
                            loader.set_description(
                                '[Train] epoch: {:04} [{}] {}'.format(epoch, tt.task_name,
                                                                      tt.eval_metrics))
                    tt.optimizer.zero_grad()
                    local_loss.backward()
                    tt.optimizer.step()
                    tt.scheduler.step()
                    del t_outputs, local_loss

                tt.criteria.reset()
                self.optimizer.zero_grad()
                final_state = tt.model.state_dict()

                delta_theta = {
                    k: outputs[k] - final_state[k].to(outputs[k].device) for k in outputs.keys()
                }

                torch.autograd.backward(list(outputs.values()), grad_tensors=list(delta_theta.values()))
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 50)
                self.optimizer.step()
                self.optimizer.zero_grad()
                # self.scheduler.step()

        self.model.zero_grad(set_to_none=True)
        tt.optimizer.state = collections.defaultdict(dict)
        tt.model.zero_grad(set_to_none=True)
        tt.eval_metrics.reset()
        torch.cuda.empty_cache()

    def eval(self, task: int, epoch: int,
             return_results: bool = False) -> Union[Metric, Tuple[Metric, Tensor]]:
        target_trainer = self.target_trainer[task]

        res_dict = {}
        for task_name, loader in target_trainer.val_loader.items():
            self.set_eval([self.model, target_trainer.model])

            loader = tqdm(loader, total=len(loader)) if self.args.show_progress else loader

            target_trainer.eval_metrics.reset()
            results = []
            with torch.no_grad():
                for batch_idx, batch in enumerate(loader):
                    batch = self.preprocessing(batch)
                    x = batch[0]
                    y = batch[1]
                    z = batch[2]

                    pattern = self.summarizer(z, task_name=[target_trainer.task_name])
                    pattern = self.tokenizer(pattern, padding='max_length', max_length=128, truncation=True,
                                             return_tensors='pt')
                    pattern = {k: v.to(self.args.device) for k, v in pattern.items()}

                    outputs = self.model(**pattern, task_name=target_trainer.task_name)

                    d = target_trainer.model.state_dict()
                    d.update(outputs)
                    target_trainer.model.load_state_dict(d)
                    t_outputs = target_trainer.model(**x).logits
                    target_trainer.eval_metrics.update(t_outputs, y)
                    results.append(t_outputs)

                    if self.args.show_progress:
                        loader.set_description(
                            '[Eval] epoch: {:04} [{}] {}'.format(epoch, task_name, target_trainer.eval_metrics))

            results = torch.cat(results).to(self.args.device).softmax(dim=-1).squeeze()
            res_dict[task_name] = results
        return (self.eval_metrics, res_dict) if return_results else self.eval_metrics

    def pred(self, task: int) -> Dict[str, Tensor]:
        target_trainer = self.target_trainer[task]
        res_dict = {}
        for task_name, loader in target_trainer.test_loader.items():
            self.set_eval([self.model, target_trainer.model])

            loader = tqdm(loader, total=len(loader)) if self.args.show_progress else loader
            results = []
            with torch.no_grad():
                for _, inst in enumerate(target_trainer.train_loader):
                    inst = self.preprocessing(inst)
                    z = inst[2]

                    pattern = self.summarizer(z, task_name=[target_trainer.task_name])
                    pattern = self.tokenizer(pattern, padding='max_length', max_length=128, truncation=True,
                                             return_tensors='pt')
                    pattern = {k: v.to(self.args.device) for k, v in pattern.items()}

                    outputs = self.model(**pattern, task_name=target_trainer.task_name)
                    d = target_trainer.model.state_dict()
                    d.update(outputs)
                    target_trainer.model.load_state_dict(d)

                    for batch_idx, batch in enumerate(loader):
                        batch = self.preprocessing(batch)
                        x = batch[0]
                        outputs = target_trainer.model(**x).logits
                        results.append(outputs)
                    break
            results = torch.cat(results).to(self.args.device)
            if results.shape[-1] > 1:
                results = results.softmax(dim=-1).argmax(dim=-1)
            res_dict[task_name] = results.squeeze()
        return res_dict

    def exec(self):
        epoch_offset = 0
        if self.args.resume_from_ckpt is not None:
            raise NotImplementedError

        best_result = None
        res = {}
        for epoch in range(epoch_offset, self.args.n_epochs):
            task_list = np.random.permutation(len(self.target_trainer))
            # task_list = range(len(self.target_trainer))
            if self.args.do_train:
                for task in task_list:
                    self.train(task, epoch)
                self.scheduler.step()

            if self.args.do_eval and epoch % self.args.evaluation_every == 0:
                for task in task_list:
                    self.eval(task, epoch, False)

            if self.args.save_model:
                result = None
                if self.save_best is None or self.save_best(result, best_result):
                    best_result = result
                    logger.info('Saving model...')
                    res = {
                        'epoch': epoch,
                        'optim': self.optimizer.state_dict(),
                        'sched': self.scheduler.state_dict(),
                        'state_dict_dir': os.path.join(self.args.output_dir, f'{self.task_name}.pt'),
                    }
                    self._save_checkpoint(res)
                    logger.info(f'Best is saved at epoch {epoch}')

        if self.args.save_model and epoch_offset < self.args.n_epochs:
            self._load_checkpoint()

        if self.args.do_test:
            for i in range(len(self.target_trainer)):
                res_dict = self.pred(i)
                for t, results in res_dict.items():
                    results = results.clone().detach().cpu().numpy()

                    with open(os.path.join(self.args.output_dir, f'{t}.tsv'), 'w') as f:
                        print(f'index\tprediction', file=f)
                        for j in range(results.shape[0]):
                            print(f'{j}\t{results[j]}', file=f)

        if self.args.save_model:
            try:
                ckpt = self._load_checkpoint()
                ckpt['epoch'] = self.args.n_epochs
                self._save_checkpoint(ckpt)
            except FileNotFoundError:
                pass
        return res


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def get_dataloader(dataset: Dataset, is_train: bool, is_distributed: bool, kwargs: Dict):
    if is_distributed:
        return DataLoader(dataset, shuffle=False, **kwargs, sampler=DistributedSampler(dataset))
    else:
        return DataLoader(dataset, shuffle=is_train, **kwargs)


def collect_parameters(model: nn.Module, target_list: List[str]) -> Tuple[List[Any], Dict[str, torch.Size]]:
    parameters = []
    named_parameters = {}
    for t in target_list:
        parameters.extend(getattr(model, t).parameters())
        named_parameters.update({f'{t}.{k}': v.shape for k, v in getattr(model, t).named_parameters()})

    return parameters, named_parameters
