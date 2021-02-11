import torch
import os
import random
import numpy as np
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


def getfminfmax(i):
    if i == 0:
        return 5345, 9080
    elif i == 1:
        return 3457, 6211
    elif i == 2:
        return 369, 3343
    elif i == 3:
        return 884, 3138
    elif i == 4:
        return 2085, 4675
    elif i == 5:
        return 4315, 12358
    elif i == 6:
        return 455, 4880
    elif i == 7:
        return 4230, 12563
    elif i == 8:
        return 3372, 6109
    elif i == 9:
        return 712, 6314
    elif i == 10:
        return 807, 11924
    elif i == 11:
        return 1595, 6277
    elif i == 12:
        return 455, 3650
    elif i == 13:
        return 26, 1601
    elif i == 14:
        return 2256, 7133
    elif i == 15:
        return 26, 1294
    elif i == 16:
        return 2771, 5084
    elif i == 17:
        return 1141, 8158
    elif i == 18:
        return 2857, 5597
    elif i == 19:
        return 197, 3138
    elif i == 20:
        return 1141, 6724
    elif i == 21:
        return 2943, 4572
    elif i == 22:
        return 9720, 15022
    else:
        return 5852, 12771