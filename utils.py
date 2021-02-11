import torch
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
import time
import sys
import math
from contextlib import contextmanager


class CFG:
    def __init__(self, args_dict):
        for k, v in args_dict.items():
            self.__setattr__(k, v)


def make_reproducible(seed=0):
    # os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.benchmark = True


def flat(l):
    return [y for x in l for y in x]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def to_logits(x):
    return np.log(np.maximum(x / (1 - x), 1e-15))


def mkdir(path_to_directory):
    p = Path(path_to_directory)
    p.mkdir(exist_ok=True, parents=True)


@contextmanager
def trace(title):
    t0 = time.time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB({sign}{delta}GB):{time.time() - t0:.1f}sec] {title} ", file=sys.stderr)


def path2id(path):
    return '.'.join(path.split('/')[-1].split('.')[:-1])


def id2path(uid, dir, ext):
    return os.path.join(dir, uid+'.'+ext)


def tomelscale(x):
    return 2595*np.log10(1+x/700)


def frommelscale(x):
    return 700*(10**(x/2595)-1)


def normalize(x):
    min, max = x.min(), x.max()
    return (x-min)/(max-min)


def transform(x, offset):
    return frommelscale(tomelscale(x)+offset).astype(int)


def getfminfmax(i, offset=100):
    df = pd.read_feather('fminfmax_by_sid.ft')
    return max(0, transform(df.loc[i, 'f_min'], -offset)), min(24000, transform(df.loc[i, 'f_max'], offset))
