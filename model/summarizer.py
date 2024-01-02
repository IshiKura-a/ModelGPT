import base64
import json
import re
from collections import OrderedDict as ODict
from io import BytesIO
from time import sleep
from typing import OrderedDict, Union, Any, Tuple
from enum import Enum

import numpy as np
import openai

from abc import ABC, abstractmethod
from typing import List, Dict

import pandas as pd
import torch
from PIL.Image import Image
from openai import InvalidRequestError
from torch import Tensor
from torchvision import transforms
from torchvision.datasets import ImageFolder
from ucimlrepo import fetch_ucirepo

from dataset.config import dataset_config, Datasets, get_transform


class LLMBackbone(Enum):
    turbo = 'gpt-3.5-turbo'
    gpt4 = 'gpt-4'
    vision = 'gpt-4-vision-preview'


class Instruction(Enum):
    parallel = f'./blob/inst_parallel.txt'
    serial = f'./blob/inst.txt'
    vision = f'./blob/inst_img_cls.txt'
    tabular = f'./blob/inst_tabular.txt'


class Summarizer(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, inputs: Any, **kwargs) -> List[str]:
        raise NotImplementedError


class SeqCLSSummarizer(Summarizer):
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


class ImageCLSSummarizer(Summarizer):
    inst = ''

    def __init__(self, model: LLMBackbone, inst: Instruction, class2idx: Dict[str, int]):
        super().__init__()
        self.model = model.value
        with open(f'./blob/config.json', 'r') as f:
            openai.api_key = json.load(f)['openai']
        with open(inst.value, 'r') as f:
            for line in f.readlines():
                self.inst += line

        self.class2idx = class2idx
        self.idx2class = [c for c, i in sorted(class2idx.items(), key=lambda item: item[1])]

    @staticmethod
    def img2base64(img: Image, fmt: str = 'png'):
        output_buffer = BytesIO()
        img.save(output_buffer, format=fmt)
        byte_data = output_buffer.getvalue()
        return f'data:image/{fmt};base64,{base64.b64encode(byte_data).decode("utf-8")}'

    def __call__(self, inputs: List[Tensor], **kwargs) -> List[str]:
        background = kwargs['background']
        transform = transforms.Compose([
            transforms.ToPILImage(),
            self.img2base64
        ])

        img_urls = [{
            "type": "image_url",
            "image_url": {
                "url": transform(x)
            }
        } for x in inputs[0]]
        labels = [self.idx2class[idx] for idx in inputs[1]]

        text_request = self.inst.format(background, ', '.join(labels))
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": text_request},
                *img_urls,
            ]}],
            temperature=0.4,
            max_tokens=len(inputs) * 128,
        )
        res: str = response['choices'][0]["message"]["content"]

        return res.split('\n')


class TabularSummarizer(Summarizer):
    inst: str = ''

    def __init__(self, model: LLMBackbone, inst: Instruction):
        super().__init__()
        self.model = model.value
        with open(f'./blob/config.json', 'r') as f:
            openai.api_key = json.load(f)['openai']
        with open(inst.value, 'r') as f:
            for line in f.readlines():
                self.inst += line

    def __call__(self, inputs: pd.DataFrame, outputs: pd.DataFrame, **kwargs) -> List[str]:
        # Only Mocked Version of this summarized is supported during training
        background = kwargs['background']
        data = [
            f'[input] {inputs.iloc[i, :].to_json()} [label] {outputs.iloc[i, 0]}' for i in
            range(inputs.shape[0])
        ]
        request = self.inst.format(background, '\n'.join(data))
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": request}],
            temperature=0.4,
            max_tokens=len(inputs) * 128,
        )
        res: str = response['choices'][0]["message"]["content"]

        return res.split('\n')


class MockSummarizer(Summarizer):
    def __init__(self, mock_data: str):
        super().__init__()
        with open(mock_data, 'r') as f:
            self.mock_data = json.load(f)

    def __call__(self, inputs: Any, **kwargs) -> List[str]:
        task_name = kwargs['task_name']
        return [
            np.random.choice(self.mock_data[t]) for t in task_name
        ]
        # return [
        #     self.mock_data[t][0] for t in task_name
        # ]


def main(test_module, **kwargs):
    if "seq_cls" in test_module:
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

        s = SeqCLSSummarizer(LLMBackbone.turbo, Instruction.parallel)
        pattern = re.compile(r'(\[(\w+)] ([\w ?!.,]+))')
        data = [
            [ODict({match[1]: match[2].strip() for match in pattern.findall(item)}) for item in
             batch.strip().split('\n')]
            for batch in re.split(r'\n\s*\n', data)]
        inputs = [ODict({k: [dic[k] for dic in d] for k in d[0]}) for d in data]
        res = s(inputs)
        print(res)

    if "mock" in test_module:
        m = MockSummarizer(f'./blob/description.json')
        print(m(None, task_name=['cola', 'mnli', 'qqp', 'qnli', 'cola']))

    if "img_cls" in test_module:
        domain = kwargs['domain']
        from util.trainer import get_dataloader
        config = dataset_config[Datasets.Office]
        d_train = ImageFolder(root=f'/root/data/dataset/office-31/{domain}/images',
                              transform=get_transform(config.target_size, True, None))
        i = ImageCLSSummarizer(LLMBackbone.vision, Instruction.vision, d_train.class_to_idx)

        loader_kwargs = {'batch_size': 32, 'num_workers': 4, 'pin_memory': True}
        train_loader = get_dataloader(d_train, True, False, loader_kwargs)

        bd = {
            "amazon": 'They are from amazon.com.',
            "dslr": 'They are taken by a dslr camera.',
            "webcam": 'They are taken by a webcam camera.'
        }
        background = bd[domain]
        success = 0
        with open(f'./blob/{domain}.txt', 'w') as f:
            for idx, batch in enumerate(train_loader):
                # success = False
                # while not success:
                #     try:
                #         print(i(batch, background=background))
                #         success = True
                #     except:
                #         print('Retrying...')
                #         sleep(1)
                try:
                    sent = i(batch, background=background)[0]
                    print(sent, file=f)
                    print(sent)
                    success += 1
                except:
                    pass
                if success == 50:
                    break

    if "tabular" in test_module:
        m = TabularSummarizer(model=LLMBackbone.turbo, inst=Instruction.tabular)
        for _d in [Datasets.Iris, Datasets.HeartDisease, Datasets.Wine, Datasets.Adult, Datasets.BreastCancer,
                   Datasets.CarEvaluation, Datasets.WineQuality, Datasets.DryBean, Datasets.Rice,
                   Datasets.BankMarketing]:
            d = fetch_ucirepo(id=_d.value)
            x = d.data.features
            y = d.data.targets

            print(f'--{_d.name}--')
            for i in range(5):
                idx = np.random.choice(y.shape[0], 8, replace=False)
                print(m(x.iloc[idx, :], y.iloc[idx, :], background=f'This is dataset {_d.name}.')[0])


if __name__ == '__main__':
    # main(test_module=["img_cls"], domain='dslr')
    # main(test_module=["img_cls"], domain='webcam')
    main(test_module=["tabular"])
