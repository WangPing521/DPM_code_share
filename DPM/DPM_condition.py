import torch
from tqdm import tqdm
from DPM_models.DPM_original import DDPM
from DPM_models.utils import extract, default, noise_like


class DDPM_condition(DDPM):
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
        DDPM.__init__(self, model, timesteps, img_size, loss_type, beta_schedule, log_every_t, v_posterior, channels, save_path)

    def diffusion_loss(self, x, t, context, noise=None):
        noise = default(noise, lambda: torch.randn_like(x))
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        model_out = self.model(x_noisy, t, context)

        loss = self.loss_fn(model_out, noise)
        return loss

    def forward(self, x, context, *args, **kwargs):
        n, c, h, w = x.shape
        assert h == w == self.img_size, "img_size should be consistent with h and w"

        t = torch.randint(1, self.timesteps, (n,), device=x.device).long()
        loss = self.diffusion_loss(x, t, context, *args, **kwargs)

        return loss

    # reverse stage: p-sample
    #todo: reverse timestep=sef-defined
    @torch.no_grad()
    def p_sample_loop(self, img, context, return_intermediates=False, noise=None):
        device = self.betas.device
        n, c, h, w = img.shape
        img_0 = torch.randn(img.shape, device=device)
        intermediates = [img_0]
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling iters', total=self.timesteps):
            img_0 = self.p_sample(
                img_0,
                torch.full((n,), i, device=device, dtype=torch.long),
                context,
            )
            if i % self.log_every_t == 0 or i == self.timesteps - 1:
                intermediates.append(img_0)
        if return_intermediates:
            return img_0, intermediates
        return img_0

    @torch.no_grad()
    def p_sample(self, x, t, context, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, context=context, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    def p_mean_variance(self, x, t, context, clip_denoised: bool):
        model_out = self.model(x, t, context)

        x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample_inference(self, img, context, sample_iters, return_intermediates=False):
        device = self.betas.device
        b = img.shape[0]
        intermediates = [img]
        for i in tqdm(reversed(range(0, sample_iters)), desc='Sampling iters', total=sample_iters):
            img = self.p_sample(
                img,
                torch.full((b,), i, device=device, dtype=torch.long),
                context,
            )
            if i % self.log_every_t == 0 or i == self.timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

