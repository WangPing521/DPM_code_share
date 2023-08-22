from pathlib import Path

from torch import nn
import torch
from typing import Any, Union, List
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn.modules.module import _addindent
from collections import OrderedDict, namedtuple

from general_utils.path_tool import path2Path


def record_err_msg(missing_keys: List[str], unexpected_keys: List[str], error_msgs: List[str]):
    if unexpected_keys:
        error_msgs.insert(0, 'Unexpected key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in unexpected_keys)))

    if missing_keys:
        error_msgs.insert(0, 'Missing key(s) in state_dict: {}. '.format(
            ', '.join(f'"{k}"' for k in missing_keys)))

def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure if there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)

def scheduler_to(sched, device):
    for param in sched.__dict__.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)

class _IncompatibleKeys(
    namedtuple("IncompatibleKeys", ["missing_keys", "unexpected_keys"])
):
    def __repr__(self):
        if not self.missing_keys and not self.unexpected_keys:
            return "<All keys matched successfully>"
        return super(_IncompatibleKeys, self).__repr__()

    __str__ = __repr__

class Buffer:
    """
    A buffer that can be used to store the state of a module.
    """

    def __init__(self, data=None):
        if isinstance(data, torch.nn.Module):
            raise ValueError(f"cannot wrap a Module in a Buffer, given {data.__class__.__name__}")

        if isinstance(data, torch.optim.Optimizer):
            raise ValueError(f"cannot wrap an Optimizer in a Buffer, given {data.__class__.__name__}")

        if isinstance(data, torch.optim.lr_scheduler._LRScheduler):  # noqa
            raise ValueError(f"cannot wrap a Scheduler in a Buffer, given {data.__class__.__name__}")

        self.data = data

    def __repr__(self):
        return f"{self.__class__.__name__}({self.data})"

class _TrainerBase(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self._persist_buffer: OrderedDict = OrderedDict()

    def __setattr__(self, key, value):
        if isinstance(value, Buffer):
            if "_persist_buffer" not in self.__dict__:
                raise AttributeError("cannot assign Buffer before Module.__init__() call")
            self._persist_buffer[key] = value.data
        elif "_persist_buffer" in self.__dict__ and key in self._persist_buffer:
            self.__setattr__(key, Buffer(value))  # noqa
        else:
            super().__setattr__(key, value)

    def __getattr__(self, item):
        if "_persist_buffer" in self.__dict__ and item in self._persist_buffer:
            return self._persist_buffer[item]
        else:
            return super().__getattr__(item)

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f'({key}): {mod_str}')

        for key, module in self._persist_buffer.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append(f'({key}): {mod_str}')
        lines = extra_lines + child_lines

        main_str = f'{self._get_name()}('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str

    def _optimizer_state_dict(self):
        return {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, Optimizer)}

    def _scheduler_state_dict(self):
        return {k: v.state_dict() for k, v in self.__dict__.items() if isinstance(v, _LRScheduler)}

    def _other_state_dict(self):
        return {
            k: v.state_dict() for k, v in self.__dict__.items() if
            k not in [*self._scheduler_state_dict(), *self._optimizer_state_dict()]
            and hasattr(v, 'state_dict') and callable(v.state_dict)
        }

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        super_state = super().state_dict(destination, prefix, keep_vars)
        buffer_state = self._persist_buffer.copy()
        optimizer_state = self._optimizer_state_dict()
        scheduler_state = self._scheduler_state_dict()
        other_state = self._other_state_dict()

        return OrderedDict({
            "module_state": super_state,
            "buffer_state": buffer_state,
            "optimizer_state": optimizer_state,
            "scheduler_state": scheduler_state,
            "other_state": other_state
        })

    def load_state_dict(self, state_dict: 'OrderedDict[str, Any]', strict=True):
        if "module_state" not in state_dict:
            raise ValueError("Missing module_state in state_dict")
        incompatible_keys = super().load_state_dict(state_dict["module_state"], strict)

        error_msgs = []
        buffer_dict = state_dict["buffer_state"]
        missing_keys = list(set(self._persist_buffer.keys()) - set(buffer_dict.keys()))
        unexpected_keys = list(set(buffer_dict.keys()) - set(self._persist_buffer.keys()))

        for key in self._persist_buffer.keys():
            if key in buffer_dict:
                self._persist_buffer[key] = buffer_dict[key]

        optimizer_dict = state_dict["optimizer_state"]
        missing_keys.extend(list(set(self._optimizer_state_dict()) - set(optimizer_dict)))
        unexpected_keys.extend(list(set(optimizer_dict) - set(self._optimizer_state_dict())))

        for name in self._optimizer_state_dict():
            if name in optimizer_dict:
                getattr(self, name).load_state_dict(optimizer_dict[name], )

        scheduler_dict = state_dict["scheduler_state"]
        missing_keys.extend(list(set(self._scheduler_state_dict()) - set(scheduler_dict)))
        unexpected_keys.extend(list(set(scheduler_dict) - set(self._scheduler_state_dict())))
        for name in self._scheduler_state_dict():
            if name in scheduler_dict:
                getattr(self, name).load_state_dict(scheduler_dict[name], )

        other_dict = state_dict["other_state"]
        missing_keys.extend(list(set(self._other_state_dict()) - set(other_dict)))
        unexpected_keys.extend(list(set(other_dict) - set(self._other_state_dict())))
        for name in self._other_state_dict():
            if name in other_dict:
                getattr(self, name).load_state_dict(other_dict[name])

        if strict:
            record_err_msg(missing_keys, unexpected_keys, error_msgs)
        if error_msgs:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                self.__class__.__name__, "\n\t".join(error_msgs)))

        return _IncompatibleKeys(incompatible_keys.missing_keys + missing_keys,
                                 incompatible_keys.unexpected_keys + unexpected_keys)

    def save_checkpoint(
            self, state_dict, current_epoch, save_dir=None, save_name=None
    ):
        """
        save checkpoint with adding 'epoch' and 'best_score' attributes
        :param state_dict:
        :param current_epoch:
        :return:
        """
        # save_best: bool = True if float(cur_score) > float(self._best_score) else False
        # if save_best:
        #     self._best_score = float(cur_score)
        state_dict["epoch"] = current_epoch
        # state_dict["best_score"] = float(self._best_score)
        save_dir = self._save_dir if save_dir is None else path2Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        if save_name is None:
            # regular saving
            torch.save(state_dict, str(save_dir / "last.pth"))
            # if save_best:
            #     torch.save(state_dict, str(save_dir / "best.pth"))
        else:
            # periodic saving
            torch.save(state_dict, str(save_dir / save_name))

    def inference(self, identifier="last.pth", *args, **kwargs):
        """
        Inference using the checkpoint, to be override by subclasses.
        :param args:
        :param kwargs:
        :return:
        """
        if self.checkpoint_path is None:
            self.checkpoint_path = self._save_dir
        assert Path(self.checkpoint_path).exists(), Path(self.checkpoint_path)
        assert (Path(self.checkpoint_path).is_dir() and identifier is not None) or (
                Path(self.checkpoint_path).is_file() and identifier is None
        )

        state_dict = torch.load(
            str(Path(self.checkpoint_path) / identifier)
            if identifier is not None
            else self.checkpoint_path,
            map_location=torch.device("cpu"),
        )
        self.load_state_dict(state_dict)
        self.model.to(self.device)
        # to be added
        # probably call self._eval() method.

    def to(self, device: Union[str, torch.device], **kwargs):

        for k, module in self.__dict__.items():
            if isinstance(module, Optimizer):
                optimizer_to(module, device)
            elif isinstance(module, _LRScheduler):
                scheduler_to(module, device)

        return super().to(device=device, **kwargs)