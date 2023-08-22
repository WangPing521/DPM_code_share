import argparse
from typing import List, Dict, Tuple, Optional, Any
from pprint import pprint
from functools import reduce, partial
from copy import deepcopy as dcp, deepcopy
from contextlib import contextmanager

import yaml
from easydict import EasyDict as edict

from general_utils.config_fn_tool import dictionary_merge_by_hierachy, remove_dictionary_callback, _load_yaml, \
    __config_dictionary__
from general_utils.config_fn_tool import merge_checker as _merge_checker

dType = Dict[str, Any]

class ConfigManager:
    def __init__(self, *path: str, verbose: bool = True,
                 strict: bool = False, _test_message=None) -> None:
        if isinstance(path[0], (list, tuple)):
            path = path[0]
        self._parsed_args, parsed_config_path, parsed_extra_args_list = \
            yamlArgParser().parse(_test_message)
        self._path = parsed_config_path or path
        self._configs: List[Dict] = self.load_yaml(verbose=False)
        self._parsed_args_merge_check = self.merge_check(strict=strict)
        self._merged_config = reduce(
            partial(dictionary_merge_by_hierachy, deepcopy=True, hook_after_merge=remove_dictionary_callback),
            [*self._configs, self._parsed_args]
        )
        if verbose:
            self.show_configs()
            self.show_merged_config()

    def load_yaml(self, verbose=False) -> List[Dict]:
        config_list = [{}]
        if self._path:
            config_list = [_load_yaml(p, verbose=verbose) for p in self._path]
        return config_list

    def merge_check(self, strict=True):
        try:
            _merge_checker(
                base_dictionary=reduce(partial(dictionary_merge_by_hierachy, deepcopy=True), self._configs),
                coming_dictionary=self._parsed_args
            )
        except RuntimeError as e:
            if strict:
                raise e

    @contextmanager
    def __call__(self, config=None, scope="base"):
        assert scope not in __config_dictionary__, scope
        config = self.config if config is None else config
        __config_dictionary__[scope] = config
        try:
            yield config
        finally:
            del __config_dictionary__[scope]

    @property
    def parsed_config(self):
        return edict(dcp(self._parsed_args))

    @property
    def unmerged_configs(self):
        return [edict(x) for x in dcp(self._configs)]

    @property
    def merged_config(self):
        return edict(dcp(self._merged_config))

    @property
    def config(self):
        return self.merged_config

    def show_configs(self):
        print("parsed configs:")

        for i, (n, d) in enumerate(zip(self._path, self._configs)):
            print(f">>>>>>>>>>>({i}): {n} start>>>>>>>>>")
            pprint(d)
        else:
            print(f">>>>>>>>>> end >>>>>>>>>")

    def show_merged_config(self):
        print("merged configure:")
        pprint(self.merged_config)

    @property
    def path(self) -> List[str]:
        return [str(x) for x in self._path]

class yamlArgParser:
    """
    parse command line args for yaml type.

    parsed_dict = YAMLArgParser()
    input:
    trainer.lr:!seq=[{1:2},{'yes':True}] lr.yes=0.94 lr.no=False
    output:
    {'lr': {'no': False, 'yes': 0.94}, 'trainer': {'lr': [{1: 2}, {'yes': True}]}}

    """

    def __init__(self, k_v_sep1: str = ":", k_v_sep2: str = "=", hierarchy: str = ".", type_sep: str = "!", ):
        self.__k_v_sep1 = k_v_sep1
        self.__k_v_sep2 = k_v_sep2
        self.__type_sep = type_sep
        self.__hierachy = hierarchy

    def parse(self, test_message=None) -> Tuple[dType, Optional[str], Optional[List[str]]]:
        parsed_args, base_filepath, extra_variable_list = self._setup(test_message)
        yaml_args: List[Dict[str, Any]] = [self.parse_string2flatten_dict(f) for f in parsed_args]
        hierarchical_dict_list = [self.create_dictionary_hierachy(d) for d in yaml_args]
        merged_args = self.merge_dict(hierarchical_dict_list)
        return merged_args, base_filepath, extra_variable_list

    @classmethod
    def _setup(cls, test_message: str = None) -> Tuple[List[str], Optional[str], List[str]]:
        parser = argparse.ArgumentParser(
            "Augment parser for yaml config", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        path_parser = parser.add_argument_group("config path parser")
        path_parser.add_argument(
            "--path", type=str, required=False, default=None, nargs=argparse.ZERO_OR_MORE,
            help="base config path location",
        )
        # parser.add_argument("--opt-path", type=str, default=None, required=False, nargs=argparse.ZERO_OR_MORE,
        #                     help="optional config path locations", )
        parser.add_argument("optional_variables", nargs="*", type=str, default=[""], help="optional variables")
        args, extra_variables = parser.parse_known_args(test_message)
        return args.optional_variables, args.path, extra_variables

    def parse_string2flatten_dict(self, string) -> Dict[str, Any]:
        """
        support yaml parser of type:
        key:value
        key=value
        key:!type=value
        to be {key:value} or {key:type(value)}
        where `:` is the `sep_1`, `=` is the `sep_2` and `!` is the `type_sep`
        :param string: input string
        :param sep_1:
        :param sep_2:
        :param type_sep:
        :return: dict
        """
        if string == "" or len(string) == 0:
            return {}

        if self.__type_sep in string:
            # key:!type=value
            # assert sep_1 in string and sep_2 in string, f"Only support key:!type=value, given {string}."
            # assert string.find(sep_1) < string.find(sep_2), f"Only support key:!type=value, given {string}."
            string = string.replace(self.__k_v_sep1, ": ")
            string = string.replace(self.__k_v_sep2, " ")
            string = string.replace(self.__type_sep, " !!")
        else:
            # no type here, so the input should be like key=value or key:value
            # assert (sep_1 in string) != (sep_2 in string), f"Only support a=b or a:b type, given {string}."
            string = string.replace(self.__k_v_sep1, ": ")
            string = string.replace(self.__k_v_sep2, ": ")

        return yaml.safe_load(string)

    @staticmethod
    def create_dictionary_hierachy(k_v_dict: Dict[str, Any]) -> Dict[str, Any]:
        if k_v_dict is None or len(k_v_dict) == 0:
            return {}
        if len(k_v_dict) > 1:
            raise RuntimeError(k_v_dict)

        key = list(k_v_dict.keys())[0]
        value = k_v_dict[key]
        keys = sorted(key.split("."), reverse=True, key=lambda x: key.split(".").index(x))
        core = {keys[0]: deepcopy(value)}
        for k in keys[1:]:
            core = {k: core}

        return core

    @staticmethod
    def merge_dict(dict_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        args = reduce(lambda x, y: dictionary_merge_by_hierachy(x, y, deepcopy=True), dict_list)
        return args
