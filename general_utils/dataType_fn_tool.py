import collections
import numbers
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import TypeVar, Set, Iterable, Callable, List, Any
from pathlib import Path
import numpy as np
from functools import reduce
from operator import and_

T_path = TypeVar("T_path", str, Path)
A = TypeVar("A")
B = TypeVar("B")
T = TypeVar("T", Tensor, np.ndarray)

def is_np_array(val):
    """
    Checks whether a variable is a numpy array.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy array. Otherwise False.

    """
    # using np.generic here via isinstance(val, (np.ndarray, np.generic)) seems to also fire for scalar numpy values
    # even though those are not arrays
    return isinstance(val, np.ndarray)

def is_np_scalar(val):
    """
    Checks whether a variable is a numpy scalar.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    out : bool
        True if the variable is a numpy scalar. Otherwise False.

    """
    # Note that isscalar() alone also fires for thinks like python strings
    # or booleans.
    # The isscalar() was added to make this function not fire for non-scalar
    # numpy types. Not sure if it is necessary.
    return isinstance(val, np.generic) and np.isscalar(val)

def uniq(a: Tensor) -> Set:
    """
    return unique element of Tensor
    Use python Optimized mode to skip assert statement.
    :rtype set
    :param a: input tensor
    :return: Set(a_npized)
    """
    return set([x.item() for x in a.unique()])

def sset(a: Tensor, sub: Iterable) -> bool:
    """
    if a tensor is the subset of the other
    :param a:
    :param sub:
    :return:
    """
    return uniq(a).issubset(sub)

def simplex(t: Tensor, axis=1) -> bool:
    """
    check if the matrix is the probability distribution
    :param t:
    :param axis:
    :return:
    """
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones, rtol=1e-4, atol=1e-4)

def one_hot(t: Tensor, axis=1) -> bool:
    """
    check if the Tensor is one hot.
    The tensor shape can be float or int or others.
    :param t:
    :param axis: default = 1
    :return: bool
    """
    return simplex(t, axis) and sset(t, [0, 1])

def class2one_hot(seg: Tensor, C: int, class_dim: int = 1) -> Tensor:
    """
    make segmentation mask to be onehot
    """
    assert sset(seg, list(range(C)))

    return F.one_hot(seg.long(), C).moveaxis(-1, class_dim)

def probs2one_hot(probs: Tensor, class_dim: int = 1) -> Tensor:
    C = probs.shape[class_dim]
    assert simplex(probs, axis=class_dim)
    res = class2one_hot(probs2class(probs, class_dim=class_dim), C, class_dim=class_dim)
    assert res.shape == probs.shape
    assert one_hot(res, class_dim)
    return res

def probs2class(probs: Tensor, class_dim: int = 1) -> Tensor:
    assert simplex(probs, axis=class_dim)
    res = probs.argmax(dim=class_dim)
    return res

def average_list(input_list):
    return sum(input_list) / len(input_list)

def average_iter(a_list):
    return sum(a_list) / float(len(a_list))

def iter_average(input_iter: Iterable):
    return sum(input_iter) / len(tuple(input_iter))

def assert_list(func: Callable[[A], bool], Iters: Iterable) -> bool:
    """
    List comprehensive assert for a function and a list of iterables.
    >>> assert assert_list(simplex, [torch.randn(2,10)]*10)
    :param func: assert function
    :param Iters:
    :return:
    """
    return reduce(and_, [func(x) for x in Iters])

def is_map(value):
    return isinstance(value, collections.abc.Mapping)

def map_(fn: Callable[[A], B], iter: Iterable[A]) -> List[B]:
    return list(map(fn, iter))

def id_(x):
    return x

def _is_tensor(tensor) -> bool:
    """
    return bool indicating if an input is a tensor of numpy or torch.
    """
    if torch.is_tensor(tensor):
        return True
    if isinstance(tensor, np.ndarray):
        return True
    return False


def _is_iterable_tensor(tensor) -> bool:
    """
    return bool indicating if an punt is a list or a tuple of tensor
    """
    from collections.abc import Iterable

    if isinstance(tensor, Iterable):
        if len(tensor) > 0:
            if _is_tensor(tensor[0]):
                return True
    return False

def _empty_iterator(tensor) -> bool:
    """
    check if a list (tuple) is empty
    """
    from collections.abc import Iterable

    if isinstance(tensor, Iterable):
        if len(tensor) == 0:
            return True
    return False

def is_iterable(val):
    """
    Checks whether a variable is iterable.

    Parameters
    ----------
    val
        The variable to check.

    Returns
    -------
    bool
        True if the variable is an iterable. Otherwise False.

    """
    return isinstance(val, collections.abc.Iterable)

def to_float(value):
    if torch.is_tensor(value):
        return float(value.item())
    elif type(value).__module__ == "numpy":
        return float(value.item())
    elif type(value) in (float, int, str):
        return float(value)
    elif isinstance(value, collections.Mapping):
        return {k: to_float(o) for k, o in value.items()}
    elif isinstance(value, (tuple, list, collections.UserList)):
        return [to_float(o) for o in value]
    else:
        raise TypeError(f"{value.__class__.__name__} cannot be converted to float.")

def to_numpy(tensor):
    if (
        is_np_array(tensor)
        or is_np_scalar(tensor)
        or isinstance(tensor, numbers.Number)
    ):
        return tensor
    elif torch.is_tensor(tensor):
        return tensor.cpu().detach().numpy()
    elif isinstance(tensor, collections.Mapping):
        return {k: to_numpy(o) for k, o in tensor.items()}
    elif isinstance(tensor, (tuple, list, collections.UserList)):
        return [to_numpy(o) for o in tensor]
    else:
        raise TypeError(f"{tensor.__class__.__name__} cannot be convert to numpy")

def to_torch(ndarray):
    if torch.is_tensor(ndarray):
        return ndarray
    elif type(ndarray).__module__ == "numpy":
        return torch.from_numpy(ndarray)
    elif isinstance(ndarray, numbers.Number):
        return torch.tensor(ndarray)
    elif isinstance(ndarray, collections.Mapping):
        return {k: to_torch(o) for k, o in ndarray.items()}
    elif isinstance(ndarray, (tuple, list, collections.UserList)):
        return [to_torch(o) for o in ndarray]
    else:
        raise ValueError("Cannot convert {} to torch tensor".format(type(ndarray)))

def flatten_dict(dictionary, parent_key="", sep="_"):
    items = []
    for k, v in dictionary.items():
        new_key = parent_key + sep + k if parent_key else k
        if is_map(v):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def prune_dict(dictionary: dict, ignore="_"):
    for k, v in dictionary.copy().items():
        if isinstance(v, dict):
            prune_dict(v, ignore)
        else:
            if k.startswith(ignore):
                del dictionary[k]

def allow_extension(path: str, extensions: List[str]) -> bool:
    try:
        return Path(path).suffixes[0] in extensions
    except:  # noqa
        return False

def is_float(v):
    """if v is a scalar"""
    try:
        float(v)
        return True
    except (ValueError, TypeError):
        return False
def _num2str(v):
    """convert a scalar to float, in order to display"""
    v = float(v)
    if abs(float(v)) < 0.01 or abs(float(v)) >= 99:
        return f"{v:.2e}"
    return f"{v:.3f}"

def _leafitem2str(v):
    if is_float(v):
        return _num2str(v)
    return f"{v}"

def _generate_pair(k, v):
    """generate str for non iterable k v"""
    return f"{k}:{_leafitem2str(v)}"

def _dict2str(dictionary: dict):
    def create_substring(k, v):
        if not is_iterable(v):
            return _generate_pair(k, v)
        else:
            return f"{k}:[" + item2str(v) + "]"

    strings = [create_substring(k, v) for k, v in dictionary.items()]
    return ", ".join(strings)
def _iter2str(item: Iterable):
    """A list or a tuple"""
    return ", ".join([_leafitem2str(x) if not is_iterable(x) else item2str(x) for x in item])

def item2str(item):
    """convert item to string in a pretty way.
        @param item: list, dictionary, set and tuple
        @return: pretty string
    """
    if isinstance(item, dict):
        return _dict2str(item)
    return _iter2str(item)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

class Identity(object):
    def __call__(self, m: Any) -> Any:
        return m
    def __repr__(self):
        return "Identify"

def norm(image: Tensor, min=0.0, max=1.0):
    min_, max_ = image.min().detach(), image.max().detach()
    image = image - min_
    image = image / (max_ - min_ + 1e-6)
    return image * (max - min) + min