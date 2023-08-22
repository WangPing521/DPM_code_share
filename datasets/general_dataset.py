import os
import re
from pathlib import Path
from typing import List, Dict, Union, Tuple, Type
from copy import deepcopy as dcp
from PIL import Image

from torch import Tensor
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from datasets.dataSamplers import PatientSampler
from datasets.dataset_abstract import default_transform
from general_utils.dataType_fn_tool import allow_extension, map_, assert_list


class Dataset_abstractClass(Dataset):
    dataset_modes = ['test']
    allow_extension = [".jpg", ".png"]

    def __init__(
        self,
        folders: List[str],
        patient_pattern: str = None,
        verbose=True,
    ) -> None:
        """
       :param sub_folders: subsubfolder name of this root, usually img, gt, etc
        :param transforms: synchronized transformation for all the subfolders
        :param verbose: verbose
        """
        assert (
                len(folders) == set(folders).__len__()
        ), f"subfolders must be unique, given {folders}."
        assert assert_list(
            lambda x: isinstance(x, str), folders
        ), f"`subfolder` elements should be path, given {folders}"
        self._name: str = f"dataset"
        self._subfolders: List[str] = folders
        self._transform = default_transform(self._subfolders)
        self._verbose = verbose
        if self._verbose:
            print(f"->> Building {self._name}:\t")
        self._filenames = self._make_dataset(
            self._subfolders, verbose=verbose
        )
        self._debug = os.environ.get("PYDEBUG", "0") == "1"
        self._set_patient_pattern(patient_pattern)

    @property
    def subfolders(self) -> List[str]:
        return self._subfolders

    @property
    def verbose(self) -> bool:
        return self._verbose

    @property
    def is_debug(self) -> bool:
        return self._debug

    def get_filenames(self, subfolder_name=None) -> List[str]:
        if subfolder_name:
            return self._filenames[subfolder_name]
        else:
            return self._filenames[self.subfolders[0]]

    def __len__(self) -> int:
        if self._debug:
            return int(len(self._filenames[self.subfolders[0]]) / 10)
        return int(len(self._filenames[self.subfolders[0]]))

    def __getitem__(self, index) -> Tuple[List[Tensor], str]:
        img_list, filename_list = self._getitem_index(index)
        assert img_list.__len__() == self.subfolders.__len__()
        # make sure the filename is the same image
        assert (
            set(map_(lambda x: Path(x).stem, filename_list)).__len__() == 1
        ), f"Check the filename list, given {filename_list}."
        filename = Path(filename_list[0]).stem
        img_list = self._transform(*img_list)
        return img_list, filename

    def _getitem_index(self, index):
        img_list = [
            Image.open(self._filenames[subfolder][index])
            for subfolder in self.subfolders
        ]
        filename_list = [
            self._filenames[subfolder][index] for subfolder in self.subfolders
        ]
        return img_list, filename_list

    def _set_patient_pattern(self, pattern):
        """
        This set patient_pattern using re library.
        :param pattern:
        :return:
        """
        assert isinstance(pattern, str), pattern
        self._pattern = pattern
        self._re_pattern = re.compile(self._pattern)

    def _get_group_name(self, path: Union[Path, str]) -> str:
        if not hasattr(self, "_re_pattern"):
            raise RuntimeError(
                "Calling `_get_group_name` before setting `set_patient_pattern`"
            )
        if isinstance(path, str):
            path = Path(path)
        try:
            group_name = self._re_pattern.search(path.stem).group(0)
        except AttributeError:
            raise AttributeError(
                f"Cannot match pattern: {self._pattern} for path: {str(path)}"
            )
        return group_name

    def get_group_list(self):

        return sorted(
            list(
                set(
                    [
                        self._get_group_name(filename)
                        for filename in self.get_filenames()
                    ]
                )
            )
        )

    @classmethod
    def _make_dataset(
        cls, folders: List[str], verbose=True
    ) -> Dict[str, List[str]]:
        # folders [img_folder, gt_folder]
        for subfolder in folders:
            assert (
                Path(subfolder).exists()
                and Path(subfolder).is_dir()
            )

        items = [
            os.listdir(Path(subfoloder))
            for subfoloder in folders
        ]
        # clear up extension
        items = sorted(
            [
                [x for x in item if allow_extension(x, cls.allow_extension)]
                for item in items
            ]
        )
        assert set(map_(len, items)).__len__() == 1, map_(len, items)

        imgs = {}
        for subfolder, item in zip(folders, items):
            imgs[subfolder] = sorted(
                [os.path.join(subfolder, x_path) for x_path in item]
            )
        assert (
            set(map_(len, imgs.values())).__len__() == 1
        ), "imgs list have component with different length."

        for subfolder in folders:
            if verbose:
                print(f"found {len(imgs[subfolder])} images in {subfolder}\t")
        return imgs

class DatasetInterfaceClass:

    def __init__(
            self,
            DataClass: Type[Dataset_abstractClass],
            root_dir: str,
            gt_dir: str,
            seed: int = 0,
            verbose: bool = True,
    ) -> None:
        super().__init__()
        self.DataClass = DataClass
        self.root_dir = root_dir
        self.seed = seed
        self.verbose = verbose

    def compile_dataloader_params(
            self,
            batch_size: int = 4,
            shuffle: bool = False,
            num_workers: int = 1,
            pin_memory: bool = True,
            drop_last=False,
    ):

        self.dataloader_params = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
        }

    def DataLoaders(self):

        _dataloader_params = dcp(self.dataloader_params)

        test_set = self._create_datasets()

        # val_loader and test_dataloader
        _dataloader_params.update({"shuffle": False})
        test_loader = (
            DataLoader(test_set, **_dataloader_params)
        )
        return test_loader


    def _create_datasets(
            self,
    ) -> Tuple[
        Dataset_abstractClass,
    ]:
        raise NotImplementedError

    def _grouped_dataloader(
            self,
            dataset: Dataset_abstractClass,
            use_infinite_sampler: bool = False,
            **dataloader_params: Dict[str, Union[int, float, bool]],
    ) -> DataLoader:
        """
        return a dataloader that requires to be grouped based on the reg of patient's pattern.
        :param dataset:
        :param shuffle:
        :return:
        """
        dataloader_params = dcp(dataloader_params)
        batch_sampler = PatientSampler(
            dataset=dataset,
            grp_regex=dataset._re_pattern,
            shuffle=dataloader_params.get("shuffle", False),
            verbose=self.verbose,
            infinite_sampler=True if use_infinite_sampler else False,
        )
        # having a batch_sampler cannot accept batch_size > 1
        dataloader_params["batch_size"] = 1
        dataloader_params["shuffle"] = False
        dataloader_params["drop_last"] = False
        return DataLoader(dataset, batch_sampler=batch_sampler, **dataloader_params)


class General_testset(Dataset_abstractClass):
    def __init__(self, *, folders: List[str], patient_pattern: str) -> None:
        super().__init__(folders=folders, patient_pattern=patient_pattern)

class General_Interface(DatasetInterfaceClass):
    def __init__(
            self,
            root_dir=Path('../../Daily_Research/Diffusion_projects/Visualizations/DiffusionMR_TestTime'),
            gt_dir=f'../../Daily_Research/Diffusion_projects/GT_data/CT/test/gt',
            verbose: bool = True,
    ) -> None:
        super().__init__(
            General_testset,
            root_dir,
            verbose,
        )
        self.img_dir = str(root_dir)
        self.gt_dir = gt_dir

    def _create_datasets(self,):
        test_set = self.DataClass(
            folders=[self.img_dir, self.gt_dir],
            patient_pattern=r"ct_train_\d+"
        )
        return test_set
