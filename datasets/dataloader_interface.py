from typing import *

from datasets.dataSamplers import InfiniteRandomSampler, PatientSampler
from datasets.dataset_abstract import MedicalImageSegmentationDataset
from copy import deepcopy as dcp
from torch.utils.data import DataLoader

class MedicalDatasetInterface:

    def __init__(
            self,
            DataClass: Type[MedicalImageSegmentationDataset],
            root_dir: str,
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

        try:
            print('[UDA: source]:labeled source data, and unlabeled target data')
            train_set, val_set, test_set = self._create_datasets()
        except:
            print('[Semi] Small amount of labeled target data, and large amount of unlabeled target')
            lab_set, unlab_set, val_set, test_set = self._create_datasets()

        # val_loader and test_dataloader
        _dataloader_params.update({"shuffle": False})
        val_loader = (
            DataLoader(val_set, **_dataloader_params)
        )
        test_loader = (
            DataLoader(test_set, **_dataloader_params)
        )

        _dataloader_params.update({"shuffle": True})
        try:
            print('Do not split lab and unlab')
            train_loader = (
                DataLoader(
                    train_set,
                    sampler=InfiniteRandomSampler(train_set, shuffle=_dataloader_params.get("shuffle")),
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle"},
                )
            )
            del _dataloader_params
            return train_loader, val_loader, test_loader

        except:
            print('SSDA target loader: lab_loader and unlab_loader')
            lab_loader = (
                DataLoader(
                    lab_set,
                    sampler=InfiniteRandomSampler(lab_set, shuffle=_dataloader_params.get("shuffle")),
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle"},
                )
            )
            unlab_loader = (
                DataLoader(
                    unlab_set,
                    sampler=InfiniteRandomSampler(unlab_set, shuffle=_dataloader_params.get("shuffle")),
                    **{k: v for k, v in _dataloader_params.items() if k != "shuffle"},
                )
            )
            del _dataloader_params
            return lab_loader, unlab_loader, val_loader, test_loader

    def _create_datasets(
            self,
    ) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset
    ]:
        raise NotImplementedError

    def _grouped_dataloader(
            self,
            dataset: MedicalImageSegmentationDataset,
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
