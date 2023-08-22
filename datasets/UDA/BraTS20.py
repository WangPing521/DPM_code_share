from pathlib import Path
from typing import Tuple, List
from datasets.dataloader_interface import MedicalDatasetInterface
from datasets.dataset_abstract import MedicalImageSegmentationDataset
from datasets.folderOperation_tool import downloading
from general_utils.path_tool import DATA_PATH


class FlairDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "BraTS20.zip"
    folder_name = "BraTS20"
    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=f'{str(path)}/flair', mode=mode, sub_folders=sub_folders, patient_pattern=patient_pattern)

class flairInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            FlairDataset,
            root_dir,
            verbose,
        )

    def _create_datasets(self) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )

        return train_set, val_set, test_set


class t1ceDataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "BraTS20.zip"
    folder_name = "BraTS20"

    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=f'{str(path)}/t1ce', mode=mode, sub_folders=sub_folders,
                         patient_pattern=patient_pattern)


class t1ceInterface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            t1ceDataset,
            root_dir,
            verbose,
        )

    def _create_datasets(self) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        return train_set, val_set, test_set


class t2Dataset(MedicalImageSegmentationDataset):
    download_link = "https://drive.google.com/uc?id=1oDKm6W6wQJRFCuiavDo3hzl7Prx2t0c0"
    zip_name = "BraTS20.zip"
    folder_name = "BraTS20"

    partition_num = 7

    def __init__(self, *, root_dir: str, mode: str, sub_folders: List[str], patient_pattern: str) -> None:
        path = Path(root_dir, self.folder_name)
        downloading(path, self.folder_name, self.download_link, root_dir, self.zip_name)
        super().__init__(root_dir=f'{str(path)}/t2', mode=mode, sub_folders=sub_folders,
                         patient_pattern=patient_pattern)


class t2Interface(MedicalDatasetInterface):
    def __init__(
            self,
            root_dir=DATA_PATH,
            verbose: bool = True,
    ) -> None:
        super().__init__(
            t2Dataset,
            root_dir,
            verbose,
        )

    def _create_datasets(self) -> Tuple[
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
        MedicalImageSegmentationDataset,
    ]:
        train_set = self.DataClass(
            root_dir=self.root_dir,
            mode="train",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        val_set = self.DataClass(
            root_dir=self.root_dir,
            mode="val",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )
        test_set = self.DataClass(
            root_dir=self.root_dir,
            mode="test",
            sub_folders=["img", "gt"],
            patient_pattern=r"BraTS20_\d+"
        )

        return train_set, val_set, test_set
