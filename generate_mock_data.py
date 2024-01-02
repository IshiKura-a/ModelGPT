from collections import OrderedDict

import numpy as np
from datasets import load_dataset

from dataset.glue import key2labels
from model.summarizer import LLMBackbone, SeqCLSSummarizer, Instruction

s = SeqCLSSummarizer(LLMBackbone.turbo, Instruction.serial)
tasks = list(key2labels.keys())
tasks.remove('ax')

d = {task: load_dataset('glue', task)['train'].remove_columns(['idx']) for task in tasks}
result = {}
for task in tasks:
    res = []
    for _ in range(5):
        batch_idx = np.random.choice(np.arange(len(d)), 4)
        batch = d[task][batch_idx]
        res.append(s([batch]))
    result[task] = res
print(result)

# print(d['train'][0])
