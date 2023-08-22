import functools
import shutil
from pathlib import Path
from functools import lru_cache
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from torch import Tensor
import torch
from skimage.io import imsave
import warnings
from typing import *

from Storage_and_Meter.metric_container import SummaryWriter, __tensorboard_queue__
from general_utils.dataType_fn_tool import _empty_iterator, _is_tensor, _is_iterable_tensor

import zipfile
import os

def save_segmentations(segs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, data_idx) -> None:
    # save the segmentation maps
    (b, w, h) = segs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning)
        for seg, name in zip(segs, names):
            if data_idx in ['BraTS20']:
                save_path = Path(root, mode, name[8:11], name).with_suffix(".png") # BraTS20
            elif data_idx in['mmwhs', 'mmwhs_new']:
                save_path = Path(root, mode, name[9:13], name).with_suffix(".png") # mmwhs

            save_path.parent.mkdir(parents=True, exist_ok=True)
            imsave(str(save_path), seg.cpu().numpy().astype(np.uint8))

def save_shifted_Imgs(imgs: Tensor, names: Iterable[str], root: Union[str, Path], mode: str, data_idx) -> None:
    if isinstance(imgs, Tensor):
        (b, w, h) = imgs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            for img, name in zip(imgs, names):
                if data_idx in ['BraTS20']:
                    save_path = Path(root, mode, name[8:11], name).with_suffix(".png")  # BraTS20
                elif data_idx in ['mmwhs', 'mmwhs_new']:
                    save_path = Path(root, mode, name[9:13], name).with_suffix(".png")  # mmwhs

                save_path.parent.mkdir(parents=True, exist_ok=True)
                imsave(str(save_path), img.cpu().numpy())
    elif isinstance(imgs, List):
        imgs = torch.stack(imgs, dim=1).squeeze(2)
        n, steps, g, w = imgs.shape
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            idx = 0
            for img in imgs:
                iter = 0
                for iter_img in img:
                    if data_idx in ['BraTS20']:
                        save_path = Path(root, mode, names[idx][8:11], names[idx], f'{iter}').with_suffix(".png")  # BraTS20
                    elif data_idx in ['mmwhs', 'mmwhs_new']:
                        save_path = Path(root, mode, names[idx][9:13], names[idx], f'{iter}').with_suffix(".png")  # mmwhs

                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    imsave(str(save_path), iter_img.cpu().numpy())
                    iter = iter + 1
                idx = idx + 1


def seg_results(img, label):
    # img.shape = n h w, 1 row n colum visualization
    fig = plt.figure()
    assert img.shape == label.shape
    assert len(img.shape) == 3

    n,h,w = img.shape
    for i in range(1, n+1):
        ax = plt.subplot(1, n, i)
        img_sub = tensor2plotable(img[i-1])
        ax.imshow(img_sub, cmap="gray")
        label_sub = tensor2plotable(label[i-1])
        ax.contour(label_sub)
    return fig

def save_diffusion_imgs(imgs: Union[Tensor, List], root: Union[str, Path], cur_epoch) -> None:
    if isinstance(imgs, list):
        imgs = torch.stack(imgs, dim=1).squeeze(2)
        n, iter, g, w = imgs.shape
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            idx = 0
            for img in imgs:
                iter = 0
                for iter_img in img:
                    save_path = Path(f'runs', root, f'{cur_epoch}epoch', f'{idx}_img', f'DPM_T_{iter}').with_suffix(".png")
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    imsave(str(save_path), iter_img.cpu().numpy())
                    iter = iter + 1
                idx = idx + 1

    elif isinstance(imgs, Tensor):
        (b, w, h) = imgs.shape  # type: Tuple[int, int,int] # Since we have the class numbers, we do not need a C axis
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            idx = 0
            for img in imgs:
                save_path = Path(f'runs', root, f'{cur_epoch}epoch', f'{idx}_img', f'TO').with_suffix(".png")
                save_path.parent.mkdir(parents=True, exist_ok=True)
                imsave(str(save_path), img.cpu().numpy())
                idx = idx + 1



def save_sdfmaps(diffusion_sdf, style):
    n1, n2, h, w = diffusion_sdf.shape
    fig = plt.figure()
    for i1 in range(1, n1 + 1):
        for i2 in range(1, n2 + 1):
            ax = plt.subplot(n1, n2, (i1 - 1) * n1 + i2)
            img = diffusion_sdf[i1 - 1, i2 - 1]

            if style == 'hot':  # probability map
                im_ = ax.imshow(img, cmap="hot")
                fig.colorbar(im_, ax=ax, orientation='vertical')
            elif style == 'grey':  # features map
                img = tensor2plotable(img)
                ax.imshow(img, cmap='gray')
    plt.title(f'SDF_DPMSDF')
    return fig

def tensor2plotable(tensor) -> np.ndarray:
    if isinstance(tensor, np.ndarray):
        return tensor
    elif isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    else:
        raise TypeError(f"tensor should be an instance of Tensor, given {type(tensor)}")

def multi_slice_viewer_debug(
    img_volume: Union[Tensor, List[Tensor], Tuple[Tensor, ...]],
    *gt_volumes: Tensor,
    no_contour=False,
    block=False,
    alpha=0.2,
) -> None:
    def process_mouse_wheel(event):
        fig = event.canvas.figure
        for i, ax in enumerate(fig.axes):
            if event.button == "up":
                previous_slice(ax)
            elif event.button == "down":
                next_slice(ax)
        fig.canvas.draw()

    def process_key(event):
        fig = event.canvas.figure
        ax = fig.axes[0]
        if event.key == "j":
            previous_slice(ax)
        elif event.key == "k":
            next_slice(ax)
        fig.canvas.draw()

    def previous_slice(ax):
        img_volume = ax.img_volume
        ax.index = (ax.index - 1) if (ax.index - 1) >= 0 else 0  # wrap around using %
        ax.images[0].set_array(img_volume[ax.index])

        if ax.gt_volume is not None:
            if not no_contour:
                for con in ax.con.collections:
                    con.remove()
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con.remove()
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")

    def next_slice(ax):
        img_volume = ax.img_volume
        ax.index = (
            (ax.index + 1)
            if (ax.index + 1) < img_volume.shape[0]
            else img_volume.shape[0] - 1
        )
        ax.images[0].set_array(img_volume[ax.index])

        if ax.gt_volume is not None:
            if not no_contour:
                for con in ax.con.collections:
                    con.remove()
                ax.con = ax.contour(ax.gt_volume[ax.index])
            else:
                ax.con.remove()
                ax.con = ax.imshow(ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow")
        ax.set_title(f"plane = {ax.index}")

    ## assertion part:
    assert _is_tensor(img_volume) or _is_iterable_tensor(
        img_volume
    ), f"input wrong for img_volume, given {img_volume}."
    assert (
        _is_iterable_tensor(gt_volumes) or gt_volumes == ()
    ), f"input wrong for gt_volumes, given {gt_volumes}."
    if _is_tensor(img_volume):
        img_volume = [img_volume]
    row_num, col_num = len(img_volume), max(len(gt_volumes), 1)

    fig, axs = plt.subplots(row_num, col_num)
    if not isinstance(axs, np.ndarray):
        # lack of numpy wrapper
        axs = np.array([axs])
    axs = axs.reshape((row_num, col_num))

    for _row_num, row_axs in enumerate(axs):
        # each row
        assert len(row_axs) == col_num
        for _col_num, ax in enumerate(row_axs):
            ax.img_volume = tensor2plotable(img_volume[_row_num])
            ax.index = ax.img_volume.shape[0] // 2
            ax.imshow(ax.img_volume[ax.index], cmap="gray")
            ax.gt_volume = (
                None
                if _empty_iterator(gt_volumes)
                else tensor2plotable(gt_volumes[_col_num])
            )
            try:
                if not no_contour:
                    ax.con = ax.contour(ax.gt_volume[ax.index])
                else:
                    ax.con = ax.imshow(
                        ax.gt_volume[ax.index], alpha=alpha, cmap="rainbow"
                    )
            except:
                pass
            ax.axis("off")
            ax.set_title(f"plane = {ax.index}")

    fig.canvas.mpl_connect("key_press_event", process_key)
    fig.canvas.mpl_connect("scroll_event", process_mouse_wheel)
    plt.show(block=block)

#---------seg line----------
def get_tb_writer() -> SummaryWriter:
    if len(__tensorboard_queue__) == 0:
        raise RuntimeError(
            "`get_tb_writer` must be call after with statement of a writer"
        )
    return __tensorboard_queue__[-1]

class switch_plt_backend:

    def __init__(self, env="agg") -> None:
        super().__init__()
        self.env = env

    def __enter__(self):
        self.prev = matplotlib.get_backend()
        matplotlib.use(self.env, force=True)

    def __exit__(self, exc_type, exc_val, exc_tb):
        matplotlib.use(self.prev, force=True)

    def __call__(self, func):
        functools.wraps(func)

        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)

        return wrapper

class FeatureMapSaver:

    def __init__(self, save_dir: Union[str, Path], folder_name="vis", use_tensorboard: bool = True) -> None:
        assert Path(save_dir).exists() and Path(save_dir).is_dir(), save_dir
        self.save_dir: Path = Path(save_dir)
        self.folder_name = folder_name
        (self.save_dir / self.folder_name).mkdir(exist_ok=True, parents=True)
        self.use_tensorboard = use_tensorboard

    @switch_plt_backend(env="agg")
    def save_map(self) -> None:
        """
        Args:

        """
        # todo

    def zip(self) -> None:
        """
        Put all image folders as a zip file, in order to avoid IO things when downloading.
        """
        try:
            shutil.make_archive(str(self.save_dir / self.folder_name.replace("/", "_")), 'zip',
                                str(self.save_dir / self.folder_name))
            shutil.rmtree(str(self.save_dir / self.folder_name))
        except (FileNotFoundError, OSError, IOError) as e:
            logger.opt(exception=True, depth=1).warning(e)

    @property
    @lru_cache()
    def tb_writer(self):
        try:
            writer = get_tb_writer()
        except RuntimeError:
            writer = None
        return writer

def compress_and_delete_folder(folder_path):
    """
    Compresses a folder using the `shutil` module and deletes the original folder
    using the `os` module to release disk space.

    Parameters:
    -----------
    folder_path : str
        The path to the folder that needs to be compressed and deleted.

    Returns:
    --------
    None
    """
    # Get the name of the folder for naming the zip file
    folder_name = os.path.basename(folder_path)

    # Compress the folder into a zip file
    shutil.make_archive(folder_name, 'zip', folder_path)

    # Delete the original folder to release disk space
    shutil.rmtree(folder_path)
