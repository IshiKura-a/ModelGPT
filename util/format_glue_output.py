import os
from pathlib import Path

from dataset.glue import key2labels


def main():
    base_dir = f'/data/tangzihao/model/modelGPT/mh_in_one'
    task_name = ['cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2', 'stsb', 'wnli', 'mnli-mm', 'ax']
    output_name = ['CoLA', 'MNLI-m', 'MRPC', 'QNLI', 'QQP', 'RTE', 'SST-2', 'STS-B', 'WNLI', 'MNLI-mm', 'AX']
    Path(os.path.join(base_dir, 'output')).mkdir(exist_ok=True)
    for t, o in zip(task_name, output_name):
        with open(os.path.join(base_dir, f'{t}.tsv'), 'r') as f:
            with open(os.path.join(base_dir, 'output', f'{o}.tsv'), 'w') as out:
                print(f.readline().strip(), file=out)
                if t in ['mnli-mm', 'ax']:
                    t = 'mnli'
                for line in f.readlines():
                    idx, label = line.strip().split('\t')
                    if t in ['ax', 'mnli', 'mnli_mismatched', 'qnli', 'rte']:
                        print(f'{idx}\t{key2labels[t][int(label)]}', file=out)
                    else:
                        print(f'{idx}\t{label}', file=out)


if __name__ == '__main__':
    main()
