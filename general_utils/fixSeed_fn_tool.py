import os
import random
import torch
import numpy as np
from contextlib import contextmanager

def fix_all_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

@contextmanager
def fix_all_seed_within_context(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_support = torch.cuda.is_available()
    if cuda_support:
        torch_cuda_state = torch.cuda.get_rng_state()
        torch_cuda_state_all = torch.cuda.get_rng_state_all()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa
        if cuda_support:
            torch.cuda.set_rng_state(torch_cuda_state)  # noqa
            torch.cuda.set_rng_state_all(torch_cuda_state_all)  # noqa

@contextmanager
def fix_all_seed_for_transforms(seed):
    random_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    fix_all_seed(seed)
    try:
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_state)  # noqa
        torch.random.set_rng_state(torch_state)  # noqa

class FixRandomSeed:
    def __init__(self, random_seed: int = 0):
        self.random_seed = random_seed
        self.randombackup = random.getstate()
        self.npbackup = np.random.get_state()

    def __enter__(self):
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def __exit__(self, *_):
        np.random.set_state(self.npbackup)
        random.setstate(self.randombackup)

