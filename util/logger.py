import io
import logging

from argparse import Namespace
from typing import List, Dict

import transformers.utils.logging

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
handlers = [logging.StreamHandler()]
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT, handlers=handlers)
logger = logging.getLogger()
transformers.utils.logging.set_verbosity_error()


def print_args(args: Namespace):
    for name, value in vars(args).items():
        logger.info(f'{name}={value}')


def save_results(results: List[Dict], filename: str):
    title = list(results[0].keys())
    with open(filename, 'w') as f:
        print(','.join(title), file=f)
        for result in results:
            print(','.join([str(i) for i in result.values()]), file=f)
