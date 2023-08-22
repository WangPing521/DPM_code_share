from torch.utils.data import Sampler
import random
import re
from typing import List, Optional, Sized, Callable, Pattern, Match, Dict
from pathlib2 import Path
import numpy as np
import torch
from itertools import repeat
from datasets.dataset_abstract import MedicalImageSegmentationDataset
from general_utils.dataType_fn_tool import map_, id_
from copy import deepcopy as dcopy
from collections import Counter
from collections import defaultdict
import os
try:
    from torch._six import int_classes as _int_classes
except ImportError:
    _int_classes = int

class InfiniteRandomSampler(Sampler):

    def __init__(self, data_source, shuffle=True):
        super().__init__(data_source)
        self.data_source = data_source
        self.shuffle = shuffle

    def __iter__(self):
        if len(self.data_source) > 0:
            while True:
                yield from self.__iter_once__()
        else:
            yield from iter([])

    def __iter_once__(self):
        if self.shuffle:
            return iter(torch.randperm(len(self.data_source)).tolist())
        return iter(torch.arange(start=0, end=len(self.data_source)).tolist())

    def __len__(self):
        return len(self.data_source)

class PatientSampler(Sampler):
    def __init__(self, dataset: MedicalImageSegmentationDataset, grp_regex: str,
                 shuffle=False, verbose=True, infinite_sampler: bool = False) -> None:
        filenames: List[str] = dataset.get_filenames()
        self.grp_regex = grp_regex
        self.shuffle: bool = shuffle
        self.shuffle_fn: Callable = (
            lambda x: random.sample(x, len(x))
        ) if self.shuffle else id_
        self._infinite_sampler = infinite_sampler
        if verbose:
            print(f"Grouping using {self.grp_regex} regex")
        grouping_regex: Pattern = re.compile(self.grp_regex)

        stems: List[str] = [
            Path(filename).stem for filename in filenames
        ]  # avoid matching the extension
        matches: List[Match] = map_(grouping_regex.match, stems)
        patients: List[str] = [match.group(0) for match in matches]

        unique_patients: List[str] = sorted(list(set(patients)))
        assert len(unique_patients) < len(filenames)
        if verbose:
            print(
                f"Found {len(unique_patients)} unique patients out of {len(filenames)} images"
            )
        self.idx_map: Dict[str, List[int]] = dict(zip(unique_patients, repeat(None)))
        for i, patient in enumerate(patients):
            if not self.idx_map[patient]:
                self.idx_map[patient] = []
            self.idx_map[patient] += [i]
        assert sum(len(self.idx_map[k]) for k in unique_patients) == len(filenames)
        if verbose:
            print("Patient to slices mapping done")

    def __len__(self):
        return len(self.idx_map.keys())

    def __iter__(self):
        if not self._infinite_sampler:
            return self._one_iter()
        return self._infinite_iter()

    def _one_iter(self):
        values = list(self.idx_map.values())
        shuffled = self.shuffle_fn(values)
        return iter(shuffled)

    def _infinite_iter(self):

        while True:
            yield from self._one_iter()

class BatchSampler(Sampler):
    r"""Wraps another sampler to yield a mini-batch of indices.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``

    Example:
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=False))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
        >>> list(BatchSampler(SequentialSampler(range(10)), batch_size=3, drop_last=True))
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    """

    def __init__(self, sampler, batch_size, drop_last):
        if not isinstance(sampler, Sampler):
            raise ValueError(
                "sampler should be an instance of "
                "torch.utils.data.Sampler, but got sampler={}".format(sampler)
            )
        if (
            not isinstance(batch_size, _int_classes)
            or isinstance(batch_size, bool)
            or batch_size <= 0
        ):
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got "
                "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0 and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify :attr:`num_samples` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`. This argument
            is supposed to be specified only when `replacement` is ``True``.
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        super(RandomSampler, self).__init__(data_source)
        self.replacement = replacement
        self._num_samples = num_samples

        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

        if self._num_samples is not None and not replacement:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(self.num_samples)
            )

    @property
    def num_samples(self):
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return self.num_samples

class SequentialSampler(Sampler):
    r"""Samples elements sequentially, always in the same order.

    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):  # noqa
        self.data_source = data_source

    def __iter__(self):
        return iter(range(len(self.data_source)))

    def __len__(self):
        return len(self.data_source)

class LimitedIterationSampler(Sampler):
    """
    this is to give a limited size of batch sampler
    """

    def __init__(self, data_source: Optional[Sized], *, stop_iteration: int, shuffle: bool = True) -> None:
        super().__init__(data_source)
        self._data_source = data_source
        self._stop_iteration = stop_iteration
        self._shuffle = shuffle
        if self._shuffle is not True:
            raise NotImplementedError(self._shuffle)

    def __iter__(self):
        available_nums = np.arange(0, len(self._data_source))
        if self._shuffle:
            chosen_nums = np.random.choice(available_nums, size=self._stop_iteration, replace=True)
            return iter(chosen_nums)

    def __len__(self):
        return self._stop_iteration

class WeightedRandomSampler(Sampler):
    r"""Samples elements from ``[0,..,len(weights)-1]`` with given probabilities (weights).

    Args:
        weights (sequence)   : a sequence of weights, not necessary summing up to one
        num_samples (int): number of samples to draw
        replacement (bool): if ``True``, samples are drawn with replacement.
            If not, they are drawn without replacement, which means that when a
            sample index is drawn for a row, it cannot be drawn again for that row.

    Example:
        >>> list(WeightedRandomSampler([0.1, 0.9, 0.4, 0.7, 3.0, 0.6], 5, replacement=True))
        [0, 0, 0, 1, 0]
        >>> list(WeightedRandomSampler([0.9, 0.4, 0.05, 0.2, 0.3, 0.1], 5, replacement=False))
        [0, 1, 4, 3, 2]
    """

    def __init__(self, weights, num_samples, replacement=True):
        if (
            not isinstance(num_samples, _int_classes)
            or isinstance(num_samples, bool)
            or num_samples <= 0
        ):
            raise ValueError(
                "num_samples should be a positive integer "
                "value, but got num_samples={}".format(num_samples)
            )
        if not isinstance(replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(replacement)
            )
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples
        self.replacement = replacement

    def __iter__(self):
        return iter(
            torch.multinomial(self.weights, self.num_samples, self.replacement).tolist()
        )

    def __len__(self):
        return self.num_samples

class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

class ContrastBatchSampler(Sampler[List[int]]):
    """
    This class is going to realize the sampling for different patients and from the same patients
    `we form batches by first randomly sampling m < M volumes. Then, for each sampled volume, we sample one image per
    partition resulting in S images per volume. Next, we apply a pair of random transformations on each sampled image and
    add them to the batch
    """

    class _SamplerIterator:

        def __init__(self, scan2index, partition2index, scan_sample_num=4, partition_sample_num=1,
                     shuffle=False) -> None:
            self._group2index, self._partition2index = dcopy(scan2index), dcopy(partition2index)

            assert 1 <= scan_sample_num <= len(self._group2index.keys()), scan_sample_num
            self._scan_sample_num = scan_sample_num
            self._partition_sample_num = partition_sample_num
            self._shuffle = shuffle

        def __iter__(self):
            return self

        def __next__(self):
            batch_index = []
            cur_group_samples = random.sample(list(self._group2index.keys()), self._scan_sample_num)
            assert isinstance(cur_group_samples, list), cur_group_samples

            # for each group sample, choose at most partition_sample_num slices per partition
            for cur_group in cur_group_samples:
                available_slices_given_group = self._group2index[cur_group]
                for s_available_slices in self._partition2index.values():
                    try:
                        sampled_slices = random.sample(
                            sorted(set(available_slices_given_group) & set(s_available_slices)),
                            self._partition_sample_num
                        )
                        batch_index.extend(sampled_slices)
                    except ValueError:
                        return self.__next__()
            if self._shuffle:
                random.shuffle(batch_index)
            return batch_index

            # fixed_one_partition = random.sample(self._partition2index.keys(), k=1)[0]
            #
            # # for each group sample, choose at most partition_sample_num slices per partition
            # for cur_group in cur_group_samples:
            #     available_slices_given_group = self._group2index[cur_group]
            #     # for s_available_slices in fixed_one_partition :
            #     s_available_slices = self._partition2index[fixed_one_partition]
            #     try:
            #         sampled_slices = random.sample(
            #             sorted(set(available_slices_given_group) & set(s_available_slices)),
            #             self._partition_sample_num
            #         )
            #         batch_index.extend(sampled_slices)
            #     except ValueError:
            #         return self.__next__()
            # if self._shuffle:
            #     random.shuffle(batch_index)
            # return batch_index

    def __init__(self, dataset, scan_sample_num=4, partition_sample_num=1, shuffle=False) -> None:
        super(ContrastBatchSampler, self).__init__(data_source=dataset)
        self._dataset = dataset
        filenames = dcopy(dataset.get_filenames())
        scan2index, partition2index = defaultdict(lambda: []), defaultdict(lambda: [])

        def _get_scan_name(filename) -> str:
            return re.compile(r"\d+").findall(os.path.basename(filename))[0]

        # get total slice_num per scan
        total_scan_dict = Counter([_get_scan_name(filename) for filename in filenames])
        total_scan_dict = dict(sorted(total_scan_dict.items()))

        def _get_partition(filename):
            scan_name = _get_scan_name(filename)
            slice_num = int(re.compile(r"\d+").findall(os.path.basename(filename))[1])
            total_scan_num = total_scan_dict[scan_name]
            return min(slice_num // (total_scan_num // dataset.partition_num), dataset.partition_num-1)

        for i, filename in enumerate(filenames):
            group = _get_scan_name(filename)  # noqa
            scan2index[group].append(i)  # noqa
            partition = _get_partition(filename)  # noqa
            partition2index[partition].append(i)  # noqa

        self._scan2index = scan2index
        self._partition2index = partition2index
        self._scan_sample_num = scan_sample_num
        self._partition_sample_num = partition_sample_num
        self._shuffle = shuffle

    def __iter__(self):
        return self._SamplerIterator(self._scan2index, self._partition2index, self._scan_sample_num,
                                     self._partition_sample_num, shuffle=self._shuffle)

    def __len__(self) -> int:
        return len(self._dataset)
