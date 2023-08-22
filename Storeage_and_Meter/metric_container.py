from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from typing import Dict, List, TypeVar, Any
from torch.utils.tensorboard import SummaryWriter as _SummaryWriter

from Storage_and_Meter.strorage_fn_tool import OrderedDict2DataFrame, rename_df_columns
from general_utils.dataType_fn_tool import prune_dict, flatten_dict
from general_utils.path_tool import path2Path
from Storage_and_Meter import general_metrics as metric
from abc import ABCMeta
from pathlib import Path
from termcolor import colored
import functools
import pandas as pd
import atexit

typePath = TypeVar("typePath", str, Path)
__tensorboard_queue__ = []

class HistoricalContainer(metaclass=ABCMeta):
    """
    Aggregate historical information in a ordered dict.
    """

    def __init__(self) -> None:
        self._record_dict: "OrderedDict" = OrderedDict()
        self._current_epoch = 0

    @property
    def record_dict(self) -> OrderedDict:
        return self._record_dict

    def __getitem__(self, index):
        return self._record_dict[index]

    @property
    def current_epoch(self) -> int:
        return self._current_epoch

    def summary(self) -> pd.DataFrame:
        return OrderedDict2DataFrame(self._record_dict)

    def add(self, input_dict, epoch=None) -> None:
        if epoch:
            self._current_epoch = epoch
        self._record_dict[self._current_epoch] = input_dict
        self._current_epoch += 1

    def reset(self) -> None:
        self._record_dict = OrderedDict()
        self._current_epoch = 0

    def state_dict(self) -> Dict[str, Any]:
        return self.__dict__

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.__dict__.update(state_dict)

    def __repr__(self):
        return str(self.summary())

class MeterInterface:
    """
    meter interface only concerns about the situation in one epoch,
    without considering historical record and save/load state_dict function.
    """

    def __init__(self, default_focus="tra") -> None:
        self._group_bank: Dict[str, Dict[str, metric.Metric]] = defaultdict(OrderedDict)
        self.__group_focus = default_focus

    def register_meter(self, name: str, meter: metric.Metric):
        return self._register_meter(name=name, meter=meter, group_name=self.__group_focus)

    def _register_meter(self, *, name: str, group_name: str, meter: metric.Metric, ) -> None:
        if not isinstance(meter, metric.Metric):
            raise KeyError(meter)
        group_meter = self._group_bank[group_name]
        if name in group_meter:
            raise KeyError(f"{name} exists in {group_name}")
        group_meter[name] = meter

    def _delete_meter(self, *, name: str, group_name: str) -> None:
        meters = self._get_meters_by_group(group_name=group_name)
        if name not in meters:
            raise KeyError(name)
        del self._group_bank[group_name][name]
        if len(self._group_bank[group_name]) == 0:
            del self._group_bank[group_name]

    def delete_meter(self, name: str):
        return self._delete_meter(name=name, group_name=self.__group_focus)

    def delete_meters(self, name_list: List[str]):
        for name in name_list:
            self.delete_meter(name=name)

    def add(self, meter_name, *args, **kwargs):
        meter = self._get_meter(name=meter_name, group_name=self.__group_focus)
        meter.add(*args, **kwargs)

    def reset(self) -> None:
        for g in self.groups():
            for m in self._group_bank[g].values():
                m.reset()

    def join(self):
        for g in self.groups():
            meters = self._get_meters_by_group(g)
            for m in meters.values():
                m.join()

    def _get_meters_by_group(self, group_name: str):
        if group_name not in self.groups():
            raise KeyError(f"{group_name} not in {self.__class__.__name__}: ({', '.join(self.groups())})")
        meters: Dict[str, metric.Metric] = self._group_bank[group_name]
        return meters

    def _get_meter(self, *, name: str, group_name: str):
        meters: Dict[str, metric.Metric] = self._get_meters_by_group(group_name=group_name)
        if name not in meters:
            raise KeyError(f"{name} not in {group_name} group: ({', '.join(meters)})")
        return meters[name]

    def groups(self):
        return list(self._group_bank.keys())

    @property
    def cur_focus(self):
        return self.__group_focus

    @contextmanager
    def focus_on(self, group_name: str):
        prev_focus = self.__group_focus
        self.__group_focus = group_name
        yield
        self.__group_focus = prev_focus

    def _statistics_by_group(self, group_name: str):
        meters = self._get_meters_by_group(group_name)
        return {k: m.summary() for k, m in meters.items()}

    def statistics(self):
        """get statistics from meter_interface. ignoring the group with name starting with `_`"""
        groups = self.groups()
        for g in groups:
            if not g.startswith("_"):
                yield g, self._statistics_by_group(g)

    def __enter__(self):
        self.reset()

    def __exit__(self, *args, **kwargs):
        self.join()

    def __getitem__(self, meter_name: str) -> metric.Metric:
        return self._get_meter(name=meter_name, group_name=self.__group_focus)

class Storage(metaclass=ABCMeta):
    r""" A container that includes all the meter results.
    """

    def __init__(self, save_dir: typePath, csv_name="storage.csv") -> None:
        super().__init__()
        self.__storage = defaultdict(HistoricalContainer)
        self._csv_name = csv_name
        self._save_dir: str = str(save_dir)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.to_csv()

    def put(
        self, name: str, value: Dict[str, float], epoch=None, prefix="", postfix=""
    ):
        self.__storage[prefix + name + postfix].add(value, epoch)

    def put_group(
        self, group_name: str, epoch_result: Dict, epoch=None, sep="/",
    ):
        assert isinstance(group_name, str), group_name
        if epoch_result:
            for k, v in epoch_result.items():
                self.put(group_name + sep + k, v, epoch)

    def add_from_meter_interface(self, *, epoch: int, **kwargs):
        for k, iterator in kwargs.items():
            for g, group_result in iterator.items():
                self.put_group(group_name=k + "/" + g, epoch_result=group_result, epoch=epoch)

    def get(self, name, epoch=None):
        assert name in self.__storage, name
        if epoch is None:
            return self.__storage[name]
        return self.__storage[name][epoch]

    def summary(self) -> pd.DataFrame:
        list_of_summary = [
            rename_df_columns(v.summary(), k, "/") for k, v in self.__storage.items()
        ]
        summary = []
        if len(list_of_summary) > 0:
            summary = functools.reduce(
                lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
                list_of_summary,
            )
        return pd.DataFrame(summary)

    @property
    def meter_names(self) -> List[str]:
        return list(self.__storage.keys())

    @property
    def storage(self):
        return self.__storage

    def state_dict(self):
        return self.__storage

    def load_state_dict(self, state_dict):
        self.__storage = state_dict
        print(colored(self.summary(), "green"))

    def to_csv(self):
        path = path2Path(self._save_dir)
        path.mkdir(exist_ok=True, parents=True)
        self.summary().to_csv(str(path / self._csv_name))

class SummaryWriter(_SummaryWriter):
    def __init__(self, log_dir: str):
        super().__init__(log_dir)
        atexit.register(self.close)

    def add_scalar_with_tag(
        self, tag, tag_scalar_dict, global_step: int, walltime=None
    ):
        """
        Add one-level dictionary {A:1,B:2} with tag
        :param tag: main tag like `train` or `val`
        :param tag_scalar_dict: dictionary like {A:1,B:2}
        :param global_step: epoch
        :param walltime: None
        :return:
        """
        assert global_step is not None
        prune_dict(tag_scalar_dict)
        tag_scalar_dict = flatten_dict(tag_scalar_dict, sep=".")
        for k, v in tag_scalar_dict.items():
            # self.add_scalars(main_tag=tag, tag_scalar_dict={k: v})
            self.add_scalar(tag=f"{tag}|{k}", scalar_value=v, global_step=global_step, walltime=walltime)

    def add_scalars_from_meter_interface(self, *, epoch: int, **kwargs):
        for g, group_dictionary in kwargs.items():
            for k, v in group_dictionary.items():
                self.add_scalar_with_tag(g + "/" + k, v, global_step=epoch)

    def __enter__(self):
        __tensorboard_queue__.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            _self = __tensorboard_queue__.pop()
            assert id(_self) == id(self)
        except Exception:
            pass
        super(SummaryWriter, self).close()
