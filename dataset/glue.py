from argparse import Namespace
from typing import Callable, Any

from transformers import PreTrainedTokenizer

task2keys = {
    'cola': ('sentence', None),
    'mnli': ('premise', 'hypothesis'),
    'ax': ('premise', 'hypothesis'),
    'mrpc': ('sentence1', 'sentence2'),
    'qnli': ('question', 'sentence'),
    'qqp': ('question1', 'question2'),
    'rte': ('sentence1', 'sentence2'),
    'sst2': ('sentence', None),
    'stsb': ('sentence1', 'sentence2'),
    'wnli': ('sentence1', 'sentence2'),
}

key2labels = {
    'cola': ['unacceptable', 'acceptable'],
    'mnli': ['entailment', 'neutral', 'contradiction'],
    'mrpc': ['not_equivalent', 'equivalent'],
    'qnli': ['entailment', 'not_entailment'],
    'qqp': ['not_duplicate', 'duplicate'],
    'rte': ['entailment', 'not_entailment'],
    'sst2': ['negative', 'positive'],
    'stsb': [],
    'wnli': ['not_entailment', 'entailment'],
    'ax': ['entailment', 'neutral', 'contradiction'],
}


def get_preprocess_fn(task_name: str, tokenizer: PreTrainedTokenizer, max_seq_length: int) -> Callable[[Any, str], Any]:
    if task_name.startswith('mnli'):
        task_name = 'mnli'
    sentence1_key = task2keys[task_name][0]
    sentence2_key = task2keys[task_name][1]

    def fn(batch: Any) -> Any:
        args = (
            (batch[sentence1_key],) if sentence2_key is None else (batch[sentence1_key], batch[sentence2_key])
        )
        result = tokenizer(*args, padding='max_length', max_length=max_seq_length, truncation=True, return_tensors='pt')
        if 'label' in batch.keys():
            result['label'] = batch['label']

        # for k in result.keys():
        #     result[k] = result[k].to(device)
        return result

    return fn


def get_num_labels(task_name: str) -> int:
    if task_name.startswith('mnli'):
        return 3
    elif task_name == 'stsb':
        return 1
    else:
        return len(key2labels[task_name])

