import torch.nn.functional as F
import torch
from inspect import isfunction
import numpy as np
from skimage import segmentation as skimage_seg
from scipy.ndimage import distance_transform_edt as distance

def sqrt_linear_betas(timesteps=1000, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta ** 0.5, max_beta ** 0.5, timesteps, dtype=torch.float64) ** 2
    return betas

def cosine_betas(timesteps=1000, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alpha_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
    betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_betas(timesteps=1000, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta, max_beta, timesteps, dtype=torch.float64)
    return betas

def sqrt_betas(timesteps=1000, min_beta=1e-4, max_beta=2e-2):
    betas = torch.linspace(min_beta, max_beta, timesteps, dtype=torch.float64) ** 0.5
    return betas

def extract(noise_coeff, t, shape_size):
    b, *_ = t.shape
    out = noise_coeff.gather(-1, t)
    return out.reshape(b, *((1,) * (len(shape_size) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))

    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

def load_checkpoint_autoencoder(encoder, decoder, segDecoder, checkpoint_path):
    #todo: load trained encoder and decoder
    weight = f"{checkpoint_path}/last.pth"
    state_dict = torch.load(weight)
    encoder = self_load_state_dict(encoder, state_dict.get('module_state'), indicator='encoder')
    decoder = self_load_state_dict(decoder, state_dict.get('module_state'), indicator='decoder')
    segDecoder = self_load_state_dict(segDecoder, state_dict.get('module_state'), indicator='segdecoder')

    for (param_en, param_de, param_seg) in zip(encoder.parameters(), decoder.parameters(), segDecoder.parameters()):
        param_en.detach_()
        param_de.detach_()
        param_seg.detach_()

    return encoder, decoder, segDecoder


def self_load_state_dict(model, state_dict: 'OrderedDict[str, Any]', indicator='encoder'):
    m_keys = model.state_dict().keys()
    pre_keys = state_dict.keys()

    for name in m_keys:
        if f'{indicator}.{name}' in pre_keys:
            param = state_dict[f'{indicator}.{name}']
            model.state_dict().get(name).copy_(param)

    return model

def compute_sdf(target: np.ndarray):
    """
    Computes the SDF of a target.

    Parameters
    ----------
    target : np.ndarray
        The target.

    Returns
    -------
    np.ndarray
        The SDF of the target.
    np.array
        The SDF of the target.
    """
    b, c, h, w = target.shape
    assert set(np.unique(target).tolist()).issubset({0,1})
    target = target.astype(np.uint8)
    output_sdf = np.zeros(target.shape, dtype=np.float32)
    out_shape = target.shape
    for b in range(out_shape[0]):
        for c in range(out_shape[1]):
            posmask = target[b, c].astype(bool)
            if posmask.any():
                negdis = distance(posmask)
                posmask = ~posmask
                posdis = distance(posmask)

                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis - posdis
                sdf[boundary == 1] = 0
                output_sdf[b][c] = sdf

    return output_sdf

def canny_edge_detection(images, sigma=3, low_threshold=0.04, high_threshold=0.1):
    # Apply Gaussian blur to reduce noise
    blurred_images = gaussian_blur(images, sigma=sigma)

    # Calculate gradients using Sobel operator
    sobel_x = F.conv2d(blurred_images, torch.tensor([[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]).float().unsqueeze(1).to(blurred_images.device), padding=1)
    sobel_y = F.conv2d(blurred_images, torch.tensor([[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]).float().unsqueeze(1).to(blurred_images.device), padding=1)

    # Calculate gradient magnitude and orientation
    gradient_mag = torch.sqrt(sobel_x**2 + sobel_y**2)
    gradient_ori = torch.atan2(sobel_y, sobel_x)

    # Non-maximum suppression
    suppressed = non_max_suppression(gradient_mag, gradient_ori)

    edges = double_thresholding(suppressed, low_threshold, high_threshold)

    return edges

def gaussian_blur(images, sigma=1):
    kernel_size = int(2 * 4 * sigma + 1)
    padding = kernel_size // 2

    x = torch.arange(kernel_size).float() - padding
    gaussian_kernel = torch.exp(-(x ** 2) / (2 * sigma ** 2))

    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0).to(images.device)
    kernel = gaussian_kernel.view(1, 1, kernel_size, 1).to(images.device)

    return F.conv2d(images, kernel, padding=(padding, padding))

def non_max_suppression(gradient_mag, gradient_ori):
    # Normalize orientation to [0, 180) degrees
    orientation = (gradient_ori * 180 / torch.tensor(np.pi)) % 180

    # Convert orientation to 0, 45, 90, or 135 degrees
    angle = torch.round(orientation / 45)

    # Get neighboring pixels based on the orientation
    neighbors = [
        (1, 0),     # angle = 0
        (1, 1),     # angle = 45
        (0, 1),     # angle = 90
        (-1, 1)     # angle = 135
    ]

    # Suppress non-maximum gradient
    suppressed = torch.zeros_like(gradient_mag)
    for dx, dy in neighbors:
        neighbor_mag = F.pad(gradient_mag, (-dx, dx, -dy, dy))
        suppressed += (gradient_mag > neighbor_mag).float()

    # Preserve only maximum gradient values
    suppressed *= gradient_mag

    return suppressed

def double_thresholding(gradient_mag, low_threshold, high_threshold):
    # Normalize gradient magnitude
    gradient_mag_norm = gradient_mag / gradient_mag.max()
    # gradient_mag_norm = gradient_mag / 255

    # Apply double thresholding
    edges = torch.zeros_like(gradient_mag)
    edges[(gradient_mag_norm >= low_threshold) & (gradient_mag_norm <= high_threshold)] = 1

    return edges