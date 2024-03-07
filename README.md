<div align='center'>

<h2><a href="https://arxiv.org/abs/2402.12408">ModelGPT: Unleashing LLM's Capabilities for Tailored Model Generation</a></h2>

[Zihao Tang](https://github.com/IshiKura-a/)<sup>1</sup>, [Zheqi Lv](https://github.com/HelloZicky)<sup>1</sup>, [Shengyu Zhang](https://shengyuzhang.github.io/)<sup>1</sup>，[Fei Wu](https://mypage.zju.edu.cn/wufei)<sup>1</sup>, [Kun Kuang](https://kunkuang.github.io/)<sup>1</sup>
 
<sup>1</sup>[Zhejiang University](https://www.zju.edu.cn/english/)
</div>
Official Pytorch Implementation for the research paper titled "ModelGPT: Unleashing LLM's Capabilities for Tailored Model Generation".

## Installation
Clone this repository and install the required packages:
```shell
git clone https://github.com/IshiKura-a/ModelGPT.git
cd ModelGPT

conda create -n ModelGPT python=3.8
conda activate ModelGPT
conda install pytorch torchvision torchaudio pytorch-cuda=12.0 -c pytorch -c nvidia

pip install -r requirements.txt
```
Download datasets:
* [Office-31](https://www.cc.gatech.edu/~judy/domainadapt/)
* GLUE Benchmark: already installed by pip requirements
* Tabular Datasets: already installed by pip requirements

## Baseline
For baseline, simply run the file in the folder `baseline`. For example, to run baseline for nlp, run:
```shell
python -m baseline.glue
```

## Train
To replicate our results, run `main_lora_nlp.py`, `main_img_cls.py`, `main_tabular.py` for nlp, cv and tabular datasets individually, like:
```shell
python main_lora_nlp.py
```
Hyperparameter settings are embedded into these files. Readers can also refer to Appendix A.

## Citation
We warmly welcome any discussion in this emerging field! If you are interested in our work, you can star our project and cite our paper:
```bib
@misc{tang2024modelgpt,
      title={ModelGPT: Unleashing LLM's Capabilities for Tailored Model Generation}, 
      author={Zihao Tang and Zheqi Lv and Shengyu Zhang and Fei Wu and Kun Kuang},
      year={2024},
      eprint={2402.12408},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
