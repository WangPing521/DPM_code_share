import random

from torch import Tensor
from general_utils.fixSeed_fn_tool import fix_all_seed_for_transforms, FixRandomSeed
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as tf
from typing import Dict, Callable, Union, List, Tuple

class RisingWrapper:

    def __init__(
            self,
            *,
            geometry_transform,
            intensity_transform
    ) -> None:
        self.geometry_transform = geometry_transform
        self.intensity_transform = intensity_transform

    def __call__(self, image: Tensor, *, mode: str, seed: int):
        assert mode in ("image", "feature"), f"`mode` must be in `image` or `feature`, given {mode}."
        if mode == "image":
            with fix_all_seed_for_transforms(seed):
                if self.intensity_transform is not None:
                    image = self.intensity_transform(data=image)["data"]
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        else:
            with fix_all_seed_for_transforms(seed):
                if self.geometry_transform is not None:
                    image = self.geometry_transform(data=image)["data"]
        return image

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8

    In the other cases, tensors are returned without scaling.
    """

    def __call__(self, pic) -> torch.Tensor:
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, torch.Tensor):
            return pic
        return tf.to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + "()"
class ToLabel(object):
    """
    PIL image to Label (long) with mapping (dict)
    """

    def __init__(self, mapping: Dict[int, int] = None) -> None:
        """
        :param mapping: Optional dictionary containing the mapping.
        """
        super().__init__()
        self.mapping = mapping
        self.mapping_call = np.vectorize(lambda x: mapping[x]) if mapping else None

    def __call__(self, img: Image.Image):
        np_img = np.array(img)[None, ...].astype(np.float32)  # type: ignore
        if self.mapping_call:
            np_img = self.mapping_call(np_img)
        t_img = torch.from_numpy(np_img)
        return t_img.long()

class SequentialWrapper:
    """
    This is the wrapper for synchronized image transformation
    The idea is to define two transformations for images and targets, with randomness.
    The randomness is garanted by the same random seed
    """

    def __init__(
        self,
        img_transform: Callable = None,
        target_transform: Callable = None,
        if_is_target: Union[List[bool], Tuple[bool, ...]] = [],
    ) -> None:
        super().__init__()
        self.img_transform = img_transform if img_transform is not None else Identity()
        self.target_transform = (
            target_transform if target_transform is not None else Identity()
        )
        self.if_is_target = if_is_target

    def __call__(
        self, *imgs, random_seed=None
    ) -> List[Union[Image.Image, torch.Tensor, np.ndarray]]:
        # assert cases
        assert len(imgs) == len(
            self.if_is_target
        ), f"len(imgs) should match len(if_is_target), given {len(imgs)} and {len(self.if_is_target)}."
        # assert cases ends
        random_seed: int = int(random.randint(0, 1e8)) if random_seed is None else int(
            random_seed
        )  # type ignore

        _imgs: List[Image.Image] = []
        for img, if_target in zip(imgs, self.if_is_target):
            with FixRandomSeed(random_seed):
                _img = self._transform(if_target)(img)
            _imgs.append(_img)
        return _imgs

    def __repr__(self):
        return (
            f"img_transform:{self.img_transform}\n"
            f"target_transform:{self.target_transform}.\n"
            f"is_target: {self.if_is_target}"
        )

    def _transform(self, is_target: bool) -> Callable:
        assert isinstance(is_target, bool)
        return self.img_transform if not is_target else self.target_transform
