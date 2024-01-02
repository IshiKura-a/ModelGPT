import functools
from typing import Dict


def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def dict2str(d: Dict) -> str:
    return ' '.join([
        f'{k} = {v.item() if hasattr(v, "item") else v}' for k, v in d.items()
    ])
