import copy
from functools import partial
from inspect import isfunction

import numpy as np
import torch
import torchvision
from torch import nn
from torch.optim import Adam
from torch.utils import data
from torchvision import transforms, utils
from tqdm import tqdm

SAVE_AND_SAMPLE_EVERY = 1000
UPDATE_EMA_EVERY = 10


def default(val, d):
    if val is not None:
        return val
    if isfunction(d):
        return d()
    return d


def cycle(dl):
    while True:
        for data in dl:
            yield data


activate = torch.nn.functional.gelu


# small helper modules

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Unet(nn.Module):
    def __init__(self, dim, out_dim=None, dim_mults=(1, 2, 4, 8), groups=8, features=32):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, features, 3, padding=1)
        self.conv1 = torch.nn.Conv2d(features, 3, 3, padding=1)
        self.dense = torch.nn.Parameter(torch.empty(2, dim, dim))
        torch.nn.init.orthogonal_(self.dense[0])
        torch.nn.init.orthogonal_(self.dense[1])

    def forward(self, x, time):
        embedding = activate(time.mm(self.dense[0])).mm(self.dense[1]).unsqueeze(2).unsqueeze(2)
        return self.conv1(self.conv0(x) + embedding)


def extract(a, t, xdim):
    b = t.size()
    out = a.gather(-1, t)
    return out.view(b, *((1,) * (xdim - 1)))


def extract_sum(a, b, c, d, t, dim):
    return extract(a, t, dim) * c + extract(b, t, dim) * d


class GaussianDiffusion(nn.Module):
    def __init__(self, denoise_fn, beta_start=0.0001, beta_end=0.02, num_diffusion_timesteps=1000, loss_type='l1',
                 betas=None):
        super().__init__()
        self.denoise_fn = denoise_fn

        if betas is None:
            self.np_betas = betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps).astype(np.float64)
        else:
            self.np_betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas

        timesteps = betas.size(0)
        self.num_timesteps = int(timesteps)
        loss_type = float(loss_type[1:])
        self.loss_fn = ((lambda x: x.pow(loss_type).mean())
                        if loss_type % 2 == 0 else
                        (lambda x: x.abs().pow(loss_type).mean()))

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.ndim) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.ndim)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.ndim)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return extract_sum(self.sqrt_recip_alphas_cumprod, self.sqrt_recipm1_alphas_cumprod, x_t, -noise, t, x_t.ndim)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = extract_sum(self.posterior_mean_coef1, self.posterior_mean_coef2, x_start, x_t, t, x_t.ndim)
        posterior_variance = extract(self.posterior_variance, t, x_t.ndim)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.ndim)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t, self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x, t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, device = x.size(0), x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        if repeat_noise:
            noise = torch.randn((1, *x.size()[1:]), device=device).expand(x.size(0), *((-1,) * (x.ndim - 1)))
        else:
            noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, image_size, batch_size=16):
        return self.p_sample_loop((batch_size, 3, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, device = x1.size(0), x1.device
        t = self.num_timesteps - 1 if t is None else t

        assert x1.shape == x2.shape

        t_batched = torch.tensor(t, device=device)
        t_batched = t_batched.unsqueeze(0).expand(b, *t_batched.size())
        xt1 = self.q_sample(x1, t_batched, torch.randn_like(x1))
        xt2 = self.q_sample(x2, t_batched, torch.randn_like(x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise):
        return extract_sum(self.sqrt_alphas_cumprod, self.sqrt_one_minus_alphas_cumprod, x_start, noise, t,
                           x_start.ndim)

    def p_losses(self, x_start, t, noise):
        x_noisy = self.q_sample(x_start, t, noise)
        x_recon = self.denoise_fn(x_noisy, t)

        loss = self.loss_fn(noise - x_recon)

        return loss

    def forward(self, x):
        b, device = x.size(0), x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, torch.randn_like(x))


# trainer class

class Trainer:
    def __init__(
            self,
            diffusion_model,
            folder,
            *,
            ema_decay=0.995,
            image_size=128,
            train_batch_size=32,
            train_lr=2e-5,
            train_num_steps=100000,
            gradient_accumulate_every=2,
            fp16=False
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)

        self.image_size = image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.ds = torchvision.datasets.ImageFolder(folder,
                                                   transforms.Compose([
                                                       transforms.Resize(image_size),
                                                       transforms.RandomHorizontalFlip(),
                                                       transforms.CenterCrop(image_size),
                                                       transforms.ToTensor()
                                                   ]))
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < 2000:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, f'./model-{milestone}.pt')

    def load(self, milestone):
        data = torch.load(f'./model-{milestone}.pt')

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                data = next(self.dl).cuda()
                loss = self.model(data)
                print(f'{self.step}: {loss.item()}')
                loss.div(self.gradient_accumulate_every).backward()

            self.opt.step()
            self.opt.zero_grad()

            if self.step % UPDATE_EMA_EVERY == 0:
                self.step_ema()

            if self.step % SAVE_AND_SAMPLE_EVERY == 0:
                milestone = self.step // SAVE_AND_SAMPLE_EVERY
                all_images = self.ema_model.p_sample_loop((64, 3, self.image_size, self.image_size))
                utils.save_image(all_images, f'./sample-{milestone}.png', nrow=8)
                self.save(milestone)

            self.step += 1

        print('training completed')
