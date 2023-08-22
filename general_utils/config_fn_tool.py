import argparse
import difflib
from pathlib import Path
from pprint import pprint
from typing import Union, Dict, Any
from collections import OrderedDict
from numbers import Number
import numpy as np
import torch
from collections.abc import Mapping
from collections.abc import Iterable
from copy import deepcopy as dcopy
import os
import yaml

from general_utils.dataType_fn_tool import is_map, is_iterable
from general_utils.path_tool import path2Path, T_path


__config_dictionary__: OrderedDict = OrderedDict()
mapType=Mapping

def get_config(scope):
    return __config_dictionary__[scope]

def _load_yaml(config_path: T_path, verbose=False):
    config_path_ = path2Path(config_path)
    assert config_path_.is_file(), config_path
    return yaml_load(config_path_, verbose=verbose)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def yaml_load(yaml_path: Union[Path, str], verbose=False) -> Dict[str, Any]:
    """
    load yaml file given a file string-like file path. return must be a dictionary.
    :param yaml_path:
    :param verbose:
    :return:
    """
    yaml_path = path2Path(yaml_path)
    assert path2Path(yaml_path).exists(), yaml_path
    if yaml_path.is_dir():
        if (yaml_path / "config.yaml").exists():
            yaml_path = yaml_path / "config.yaml"
        else:
            raise FileNotFoundError(f"config.yaml does not found in {str(yaml_path)}")

    with open(str(yaml_path), "r") as stream:
        data_loaded: dict = yaml.safe_load(stream)
    if verbose:
        print(f"Loaded yaml path:{str(yaml_path)}")
        pprint(data_loaded)
    return data_loaded

def yaml_write(
        dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> str:
    save_path = path2Path(save_dir) / save_name
    path2Path(save_dir).mkdir(exist_ok=True, parents=True)
    if save_path.exists():
        if force_overwrite is False:
            save_path = (
                    save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_path), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)
    return str(save_path)

def edict2dict(item):
    if isinstance(item, (str, Number, float, np.ndarray, torch.Tensor)):
        return item
    if isinstance(item, (list, tuple)):
        return type(item)([edict2dict(x) for x in item])
    if isinstance(item, dict):
        return {k: edict2dict(v) for k, v in item.items()}

def dictionary_merge_by_hierachy(dictionary1: Dict[str, Any], new_dictionary: Dict[str, Any] = None, deepcopy=True,
                                 hook_after_merge=None):
    """
    Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into``dct``.
    :return: None
    """
    if deepcopy:
        dictionary1, new_dictionary = dcopy(dictionary1), dcopy(new_dictionary)
    if new_dictionary is None:
        return dictionary1
    for k, v in new_dictionary.items():
        if k in dictionary1 and isinstance(dictionary1[k], mapType) and isinstance(new_dictionary[k], mapType):
            dictionary1[k] = dictionary_merge_by_hierachy(dictionary1[k], new_dictionary[k], deepcopy=False)
        else:
            dictionary1[k] = new_dictionary[k]
    if hook_after_merge:
        dictionary1 = hook_after_merge(dictionary1)
    return dictionary1

def remove_dictionary_callback(dictionary, key="remove"):
    new_dictionary = dcopy(dictionary)
    for k, v in dictionary.items():
        if isinstance(v, mapType):
            new_dictionary[k] = remove_dictionary_callback(v, key)
        try:
            if v.lower() == key:
                del new_dictionary[k]
        except AttributeError:
            pass
    return new_dictionary

def extract_dictionary_from_anchor(target_dictionary: Dict, anchor_dictionary: Dict, deepcopy=True, prune_anchor=False):
    result_dict = {}

    if deepcopy:
        anchor_dictionary, target_dictionary = map(dcopy, (anchor_dictionary, target_dictionary))

    for k, v in anchor_dictionary.items():
        if k in target_dictionary:
            if not isinstance(v, mapType):
                result_dict[k] = target_dictionary[k]
            else:
                result_dict[k] = extract_dictionary_from_anchor(target_dictionary[k], anchor_dictionary[k],
                                                                deepcopy=deepcopy, prune_anchor=prune_anchor)
        elif not prune_anchor:
            result_dict[k] = anchor_dictionary[k]

    return result_dict

def dictionary2string(dictionary, parent_name_list=None, item_list=None):
    def tostring(item):
        if isinstance(item, (float,)):
            return f"{item:.7f}"
        return str(item)

    if parent_name_list is None:
        parent_name_list = []
    if item_list is None:
        item_list = []
    for k, v in dictionary.items():
        if is_map(v):
            dictionary2string(v, parent_name_list=parent_name_list + [k], item_list=item_list)
        elif isinstance(v, Iterable) and (not isinstance(v, str)):
            current_item = ".".join(parent_name_list) + f".{k}=[{','.join([tostring(x) for x in v])}]"
            item_list.append(current_item)
        else:
            current_item = ".".join(parent_name_list) + f".{k}={tostring(v)}"
            item_list.append(current_item)
    return " ".join(item_list)

def extract_params_with_key_prefix(item: Dict[str, Any], prefix: str) -> Dict:
    # if isinstance(dictionary, (str, int, float, torch.Tensor, np.ndarray)):
    #     return dictionary
    if is_map(item):
        result_dict = {}
        for k, v in item.items():
            if is_map(v):
                result_dict[k] = extract_params_with_key_prefix(v, prefix=prefix)
            elif is_iterable(v):
                result_dict[k] = [extract_params_with_key_prefix(x, prefix=prefix) for x in v]
            else:
                if k.startswith(prefix):
                    result_dict[k.replace(prefix, "")] = v

            # clean items with {}
            for _k, _v in result_dict.copy().items():
                if _v == {}:
                    del result_dict[_k]
        return result_dict
    if is_iterable(item):
        return type(item)([extract_params_with_key_prefix(x, prefix=prefix) for x in item])
    if isinstance(item, (str, Number, torch.Tensor, np.ndarray)):
        return item
    else:
        raise RuntimeError(item)

def __name_getter(dictionary: mapType, previous_name, previous_names):
    for k, v in dictionary.items():
        if previous_name == "":
            previous_names.append(k)
        else:
            previous_names.append(str(previous_name) + "." + str(k))
    for k, v in dictionary.items():
        if isinstance(v, mapType):
            __name_getter(v, str(k) if previous_name == "" else str(previous_name) + "." + str(k), previous_names, )

def merge_checker(base_dictionary, coming_dictionary):
    base_names, coming_names = [], []
    __name_getter(base_dictionary, "", base_names), __name_getter(coming_dictionary, "", coming_names)

    undesired_attributes = sorted(set(coming_names) - set(base_names))

    def create_possible_suggestion(unwanted_string: str):
        candidate_list = difflib.get_close_matches(unwanted_string, base_names, n=1)
        if len(candidate_list) > 0:
            return candidate_list[0]
        else:
            return ""

    if len(undesired_attributes) > 0:
        raise RuntimeError(
            f"\nUnwanted attributed identified compared with base config: \t"
            f"{', '.join([f'`{x}`: (possibly `{create_possible_suggestion(x)}`)' for x in undesired_attributes])}"
        )

def write_yaml(
        dictionary: Dict, save_dir: Union[Path, str], save_name: str, force_overwrite=True
) -> None:
    save_path = path2Path(save_dir) / save_name
    if save_path.exists():
        if force_overwrite is False:
            save_name = (
                    save_name.split(".")[0] + "_copy" + "." + save_name.split(".")[1]
            )
    with open(str(save_dir / save_name), "w") as outfile:  # type: ignore
        yaml.dump(dictionary, outfile, default_flow_style=False)

def set_environment(environment_dict: Dict[str, str] = None, verbose=True) -> None:
    if environment_dict:
        for k, v in environment_dict.items():
            os.environ[k] = str(v)
            if verbose:
                print(f"setting environment {k}:{v}")

