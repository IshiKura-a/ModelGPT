import json
import re
from collections import OrderedDict as ODict
from typing import OrderedDict, Union
from enum import Enum

import numpy as np
import openai

from abc import ABC, abstractmethod
from typing import List, Dict


class LLMBackbone(Enum):
    turbo = 'gpt-3.5-turbo'
    gpt4 = 'gpt-4'


class Instruction(Enum):
    parallel = f'./blob/inst_parallel.txt'
    serial = f'./blob/inst.txt'


class Summarizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, inputs: Union[OrderedDict[str, List], List[OrderedDict[str, List]]], **kwargs) -> List[str]:
        raise NotImplementedError


class LLMAPISummarizer(Summarizer):
    inst: str = ''

    def __init__(self, model: LLMBackbone, inst: Instruction):
        super().__init__()
        self.model = model.value
        with open(f'./blob/config.json', 'r') as f:
            openai.api_key = json.load(f)['openai']
        with open(inst.value, 'r') as f:
            for line in f.readlines():
                self.inst += line

    def __call__(self, inputs: Union[OrderedDict[str, List], List[OrderedDict[str, List]]], **kwargs) -> List[str]:
        if isinstance(inputs, ODict):
            inputs = [inputs]
        sub_batch_size = len(list(inputs[0].values())[0])
        request = self.inst.format('\n\n'.join(
            ['\n'.join([' '.join([f'[{k}] {v[i]}' for k, v in batch.items()]) for i in range(sub_batch_size)]) for batch
             in inputs]))

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": request}],
            temperature=0.4,
            max_tokens=len(inputs) * 128,
        )
        res: str = response['choices'][0]["message"]["content"]

        # print(res)
        return res.split('\n')


class MockSummarizer(Summarizer):
    def __init__(self, mock_data: str):
        super().__init__()
        with open(mock_data, 'r') as f:
            self.mock_data = json.load(f)

    def __call__(self, inputs: Union[OrderedDict[str, List], List[OrderedDict[str, List]]], **kwargs) -> List[str]:
        task_name = kwargs['task_name']
        return [
            np.random.choice(self.mock_data[t]) for t in task_name
        ]


def main():
    data = """
    [question1] How is the life of a math student? Could you describe your own experiences? [question2] Which level of prepration is enough for the exam jlpt5? [label] not_duplicate
    [question1] How do I control my horny emotions? [question2] How do you control your horniness? [label] duplicate
    [question1] What causes stool color to change to yellow? [question2] What can cause stool to come out as little balls? [label] not_duplicate
    [question1] What can one do after MBBS? [question2] What do i do after my MBBS ? [label] duplicate
    
    [sentence1] Judie Vivian, chief executive at ProMedica, a medical service company that helps sustain the 2-year-old Vietnam Heart Institute in Ho Chi Minh City (formerly Saigon), said that so far about 1,500 children have received treatment. [sentence2] The previous name of Ho Chi Minh City was Saigon. [label] entailment
    [sentence1] A man is due in court later charged with the murder 26 years ago of a teenager whose case was the first to be featured on BBC One's Crimewatch. Colette Aram, 16, was walking to her boyfriend's house in Keyworth, Nottinghamshire, on 30 October 1983 when she disappeared. Her body was later found in a field close to her home. Paul Stewart Hutchinson, 50, has been charged with murder and is due before Nottingham magistrates later. [sentence2] Paul Stewart Hutchinson is accused of having stabbed a girl. [label] not_entailment
    [sentence1] Britain said, Friday, that it has barred cleric, Omar Bakri, from returning to the country from Lebanon, where he was released by police after being detained for 24 hours. [sentence2] Bakri was briefly detained, but was released. [label] entailment
    [sentence1] Nearly 4 million children who have at least one parent who entered the U.S. illegally were born in the United States and are U.S. citizens as a result, according to the study conducted by the Pew Hispanic Center. That's about three quarters of the estimated 5.5 million children of illegal immigrants inside the United States, according to the study. About 1.8 million children of undocumented immigrants live in poverty, the study found. [sentence2] Three quarters of U.S. illegal immigrants have children. [label] not_entailment
    
    [sentence] hide new secretions from the parental units [label] negative
    [sentence] contains no wit , only labored gags [label] negative
    [sentence] that loves its characters and communicates something rather beautiful about human nature [label] positive
    [sentence] remains utterly satisfied to remain the same throughout [label] negative
    
    [sentence1] A plane is taking off. [sentence2] An air plane is taking off. [label] 5
    [sentence1] A man is playing a large flute. [sentence2] A man is playing a flute. [label] 3.8
    [sentence1] A man is spreading shreded cheese on a pizza. [sentence2] A man is spreading shredded cheese on an uncooked pizza. [label] 3.8
    [sentence1] Three men are playing chess. [sentence2] Two men are playing chess. [label] 2.6
    [sentence1] A man is playing the cello. [sentence2] A man seated is playing the cello. [label] 4.25
    
    [sentence1] I stuck a pin through a carrot. When I pulled the pin out, it had a hole. [sentence2] The carrot had a hole. [label] entailment
    [sentence1] John couldn't see the stage with Billy in front of him because he is so short. [sentence2] John is so short. [label] entailment
    [sentence1] The police arrested all of the gang members. They were trying to stop the drug trade in the neighborhood. [sentence2] The police were trying to stop the drug trade in the neighborhood. [label] entailment
    [sentence1] Steve follows Fred's example in everything. He influences him hugely. [sentence2] Steve influences him hugely. [label] not_entailment
    """

    s = LLMAPISummarizer(LLMBackbone.turbo, Instruction.parallel)
    pattern = re.compile(r'(\[(\w+)] ([\w ?!.,]+))')
    data = [
        [ODict({match[1]: match[2].strip() for match in pattern.findall(item)}) for item in batch.strip().split('\n')]
        for batch in re.split(r'\n\s*\n', data)]
    inputs = [ODict({k: [dic[k] for dic in d] for k in d[0]}) for d in data]
    res = s(inputs)

    m = MockSummarizer(f'./blob/description.json')
    print(m(None, task_name=['cola', 'mnli', 'qqp', 'qnli', 'cola']))


if __name__ == '__main__':
    main()
