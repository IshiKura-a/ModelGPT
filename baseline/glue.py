from typing import Any

from collections import OrderedDict as Odict

import torch
import os
import time
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from torch import optim, is_tensor, nn
from torch.optim.lr_scheduler import ConstantLR
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, \
    DistilBertForSequenceClassification

from dataset.glue import get_preprocess_fn, task2keys, key2labels, get_num_labels
from util.criteria import CrossEntropy, Accuracy, Compose, MSELoss, BCELoss
from util.logger import logger
from util.trainer import TrainingArguments, Trainer, seed_everything


def preprocess_fn(batch: Any, device: str) -> Any:
    text = {}
    labels = batch['label'].to(device)
    batch.pop('label')
    batch.pop('idx')
    to_del = []
    for k, v in batch.items():
        if not is_tensor(v):
            text[k] = v
            to_del.append(k)
        else:
            batch[k] = v.to(device)

    for k in to_del:
        batch.pop(k)

    if labels.dtype == torch.double:
        labels = labels.to(dtype=torch.float)
    return batch, labels, Odict(text)


def run(task_name: str,
        do_train: bool, do_eval: bool, do_test: bool,
        model_name_or_path: str,
        tokenizer_name_or_path: str,
        seed: int,
        output_dir: str,
        fc_only: bool = False):
    logger.info(task_name)
    seed_everything(seed)

    d = load_dataset('/root/data/dataset/glue', task_name)
    num_labels = get_num_labels(task_name)
    if num_labels == 1:
        criteria = MSELoss()
        eval_metrics = Compose([MSELoss()])
    else:
        criteria = CrossEntropy()
        eval_metrics = Compose([CrossEntropy(), Accuracy()])

    config = AutoConfig.from_pretrained(
        tokenizer_name_or_path,
        num_labels=num_labels,
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=config)
    d = d.map(
        get_preprocess_fn(task_name, tokenizer, 128),
        batched=True,
        load_from_cache_file=True,
        desc='Running tokenizer on dataset',
    )
    d.set_format("pt", columns=["input_ids", "attention_mask"], output_all_columns=True)
    # d.set_format("pt", columns=["label"], output_all_columns=True, dtype=torch.float)

    val_key = 'validation_matched' if task_name == 'mnli' else 'validation'
    test_key = 'test_matched' if task_name == 'mnli' else 'test'
    loader_kwargs = {'batch_size': 256, 'num_workers': 4, 'pin_memory': True}
    train_loader = DataLoader(d['train'], shuffle=True, **loader_kwargs) if do_train else None
    val_loader = DataLoader(d[val_key], shuffle=False, **loader_kwargs) if do_eval else None
    test_loader = DataLoader(d[test_key], shuffle=False, **loader_kwargs) if do_test else None

    args = TrainingArguments(do_train=do_train,
                             do_eval=do_eval,
                             do_test=do_test,
                             n_epochs=20,
                             output_dir=output_dir,
                             save_model=do_train)
    if not fc_only:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=r'.*[qv]_lin',
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    else:
        parameters = []
        parameters.extend(model.pre_classifier.parameters())
        parameters.extend(model.classifier.parameters())
        optimizer = optim.Adam(parameters, lr=1e-3, weight_decay=1e-4)
    scheduler = ConstantLR(optimizer)
    trainer = Trainer(args=args,
                      model=model,
                      task_name=task_name,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      test_loader=test_loader,
                      optimizer=optimizer,
                      scheduler=scheduler,
                      criteria=criteria,
                      eval_metrics=eval_metrics,
                      save_best=(lambda cur, prev: prev is None or cur < prev) if task_name == 'stsb' else (
                          lambda cur, prev: prev is None or cur['CE'] < prev['CE']),
                      preprocessing=preprocess_fn,
                      )
    trainer.exec()


def main():
    begin = time.time()
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    model = '/root/data/model/distilbert-base-uncased'
    output_dir = f'/root/data/model/distilbert_glue/lora'
    for task_name in ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli']:
        run(task_name, True, True, True, model,
            model, 2024, output_dir,
            False)

    for task_name in ['mnli_mismatched', 'ax']:
        run(task_name, False, False, True, f'{output_dir}/mnli.pt',
            model, 2024, output_dir,
            False)
    # run('stsb', True, True, False, '/root/data/model/distilbert-base-uncased',
    #     '/root/data/model/distilbert-base-uncased', 2024, f'/root/data/model/distilbert_glue/',
    #     False)
    end = time.time()
    logger.info(end - begin)


if __name__ == '__main__':
    main()
