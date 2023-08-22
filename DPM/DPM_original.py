from torch import nn
import torch
import torch.nn.functional as F
from tqdm import tqdm

from DPM.utils import linear_betas, sqrt_linear_betas, sqrt_betas, cosine_betas, extract, default, noise_like


class DDPM(nn.Module):
    def __init__(self,
                 model,
                 timesteps=1000,
                 img_size=256,
                 loss_type='L1',
                 beta_schedule='linear',
                 log_every_t=50,
                 v_posterior=0,
                 channels=1,
                 save_path='runs',
                 ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.img_size = img_size
        self.loss_type = loss_type
        self.beta_schedule = beta_schedule
        self.log_every_t = log_every_t
        self.save_path = save_path
        self.channels = channels
        self.v_posterior = v_posterior
        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        # register betas
        if self.beta_schedule  == "linear":
            betas = linear_betas(self.timesteps, min_beta=1e-4, max_beta=2e-2)
        elif self.beta_schedule == "cosine":
            betas = cosine_betas(self.timesteps, s=0.008)
        elif self.beta_schedule == "sqrt_linear":
            betas = sqrt_linear_betas(self.timesteps, min_beta=1e-4, max_beta=2e-2)
        elif self.beta_schedule == "sqrt":
            betas = sqrt_betas(self.timesteps, min_beta=1e-4, max_beta=2e-2)
        else:
            raise ValueError(f"This kind of betas scheduler is unknown.")

        register_buffer("betas", betas)

        # register alphas
        alphas = 1 - betas
        register_buffer("alphas", alphas)

        # register alphas_cumprod
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        register_buffer("alphas_cumprod", alphas_cumprod)

        # register alphas_cumprod_prev
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # register scheduler for diffusion q(x_t | x_{t-1}) and others
        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0-alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))

        # register scheduler for posterior q(x_{t-1} | x_t, x_0)
        # when v_posterior=0, it becomes betas * ((1- alpha_cumprod_prev) / (1-alpha_cumprod))
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) + self.v_posterior * betas
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(torch.max(posterior_variance, torch.Tensor([1e-20]))))

        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    @property
    def loss_fn(self):
        if self.loss_type == "L1":
            return F.l1_loss
        elif self.loss_type == 'L2':
            return F.mse_loss
        else:
            raise ValueError(F"Invalid loss type {self.loss_type}")

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def diffusion_loss(self, x, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_out, _ = self.model(x_noisy, t)

        loss = self.loss_fn(model_out, noise)
        return loss

    def forward(self, x, *args, **kwargs):
        n, c, h, w = x.shape
        assert h == w == self.img_size, "img_size should be consistent with h and w"

        t = torch.randint(1, self.timesteps, (n,), device=x.device).long()
        loss = self.diffusion_loss(x, t, *args, **kwargs)

        return loss

    # reverse stage: p-sample
    #todo: reverse timestep=sef-defined
    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        image_size = self.img_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size), return_intermediates=return_intermediates)

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling iters', total=self.timesteps):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
            )
            if i % self.log_every_t == 0 or i == self.timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, t, clip_denoised: bool):
        model_out, _ = self.model(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def p_sample_inference(self, img, sample_iters, return_intermediates=False):
        device = self.betas.device
        b = img.shape[0]
        intermediates = [img]
        for i in tqdm(reversed(range(0, sample_iters)), desc='Sampling iters', total=sample_iters):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
            )
            if i % self.log_every_t == 0 or i == self.timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

