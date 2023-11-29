import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
# from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import CIFAR10
from torch.utils.data import DataLoader
import json
from fvcore.common.config import CfgNode
from scipy import integrate
import matplotlib.pyplot as plt
import numpy as np

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def main():
    # cfg = CfgNode(load_conf('./configs/local_debug.yaml'))
    # cfg = CfgNode(load_conf('./configs/cifar10/uncond_etp_const_dpm_sde4.yaml'))
    # cfg.model.ckpt_path = '/home/huang/Downloads/mobaxterm/const-sde4-model-20.pt'
    # cfg = CfgNode(load_conf('./configs/cifar10/uncond_original_dpm.yaml'))
    # cfg.model.ckpt_path = '/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/exp/diffusion/cifar10/results_original_dpm/model-20.pt'
    # cfg.model.ckpt_path = '/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/exp/diffusion/pre_weights/const-sde2-model-6.pt'
    cfg = CfgNode(load_conf('./configs/cifar10/uncond_etp_2order_dpm_sde_ncsnpp6.yaml'))
    cfg.model.ckpt_path = '/media/huang/2da18d46-7cba-4259-9abd-0df819bb104c/exp/diffusion/cifar10/results_etp_2order_sde_ncsnpp6/model-78.pt'
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    # first_stage_cfg = model_cfg.first_stage
    # first_stage_model = AutoencoderKL(
    #     ddconfig=first_stage_cfg.ddconfig,
    #     lossconfig=first_stage_cfg.lossconfig,
    #     embed_dim=first_stage_cfg.embed_dim,
    #     ckpt_path=first_stage_cfg.ckpt_path,
    # )
    # unet_cfg = model_cfg.transmodel
    # unet = TransModel(dim=unet_cfg.dim,
    #             channels=unet_cfg.channels,
    #             encoder_layers=unet_cfg.encoder_layers,
    #             decoder_layers=unet_cfg.decoder_layers,
    #             patch_size=unet_cfg.patch_size,
    #             cond_in_dim=unet_cfg.cond_in_dim,
    #             cond_dim=unet_cfg.cond_dim,
    #             cond_dim_mults=unet_cfg.cond_dim_mults,
    #             spatial_size=unet_cfg.spatial_size,
    #             num_groups=unet_cfg.num_groups,
    #             learned_variance=unet_cfg.learned_variance,
    #             )
    if model_cfg.model_name == 'ncsnpp':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp import NCSNpp
        unet = NCSNpp(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp2':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp2 import NCSNpp
        unet = NCSNpp(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp3':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp3 import NCSNpp
        unet = NCSNpp(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp4':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp4 import NCSNpp
        unet = NCSNpp(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp5':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp5 import NCSNpp
        unet = NCSNpp(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp6':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp6 import NCSNpp
        unet = NCSNpp(unet_cfg)
    else:
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.learned_variance,
                    out_mul=unet_cfg.out_mul,
                    heads=unet_cfg.heads,
                    # cond_in_dim=unet_cfg.cond_in_dim,
                    # cond_dim=unet_cfg.cond_dim,
                    # cond_dim_mults=unet_cfg.cond_dim_mults,
                    # window_sizes1=unet_cfg.window_sizes1,
                    # window_sizes2=unet_cfg.window_sizes2,
                    )
    if model_cfg.model_type == 'linear':
        from denoising_diffusion_pytorch.etp_dpm_linear import DDPM
    elif model_cfg.model_type == 'original':
        from denoising_diffusion_pytorch.denoising_diffusion_original import GaussianDiffusion
        DDPM = GaussianDiffusion
    elif model_cfg.model_type == 'const':
        from denoising_diffusion_pytorch.etp_dpm_const import DDPM
    elif model_cfg.model_type == 'linear2':
        from denoising_diffusion_pytorch.etp_dpm_linear2 import DDPM  # without C
    elif model_cfg.model_type == 'exp':
        from denoising_diffusion_pytorch.etp_dpm_exp import DDPM
    elif model_cfg.model_type == 'exp2':
        from denoising_diffusion_pytorch.etp_dpm_exp2 import DDPM
    elif model_cfg.model_type == '2order':
        from denoising_diffusion_pytorch.etp_dpm_2order import DDPM
    elif model_cfg.model_type == '2order_rec':
        from denoising_diffusion_pytorch.etp_dpm_2order_rec import DDPM
    elif model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.etp_dpm_const_sde import DDPM
    elif model_cfg.model_type == 'const_sde2':
        from denoising_diffusion_pytorch.etp_dpm_const_sde2 import DDPM
    elif model_cfg.model_type == 'const_sde3':
        from denoising_diffusion_pytorch.etp_dpm_const_sde3 import DDPM
    elif model_cfg.model_type == 'const_sde4':
        from denoising_diffusion_pytorch.etp_dpm_const_sde4 import DDPM
    elif model_cfg.model_type == 'linear_sde':
        from denoising_diffusion_pytorch.etp_dpm_linear_sde import DDPM
    elif model_cfg.model_type == '2order_sde':
        from denoising_diffusion_pytorch.etp_dpm_2order_sde import DDPM
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    dpm = DDPM(
        model=unet,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=model_cfg.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
    ).cuda()
    # dpm.sample(batch_size=2)
    data_cfg = cfg.data
    # dataset = CIFAR10(
    #     img_folder=data_cfg.img_folder,
    #     image_size=model_cfg.image_size,
    #     augment_horizontal_flip=data_cfg.augment_horizontal_flip
    # )
    # dl = DataLoader(dataset, batch_size=data_cfg.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    batch_size = 16

    # ode_sampler = get_ode_sampler(method='RK45', model_type=model_cfg.model_type, device='cuda:0')
    # x, nfe = ode_sampler(dpm.model, (1, 3, 32, 32))
    # x.clamp_(0., 1.)

    device = torch.device('cuda:0')
    with torch.no_grad():
        image_size, channels = dpm.image_size, dpm.channels
        shape = (batch_size, channels, image_size[0], image_size[1])
        # times = torch.linspace(-1, dpm.num_timesteps, steps=10 + 1).int()
        # times = torch.tensor([-1, 180, 360, 530, 680, 810, 890, 940, 970, 990, 1000]).int()
        # times = torch.tensor([-1, 10, 20, 30, 50, 100, 200, 350, 500, 700, 1000]).int()
        # times = list(reversed(times.int().tolist()))
        # time_pairs = list(zip(times[:-1], times[1:]))
        time_steps = torch.tensor([0.01]).repeat(100)
        img = torch.randn(shape).to(device)
        # K = -1 * torch.ones_like(img)
        imgs = [img]
        Ks1 = []
        Ks2 = []
        Cs = []
        K1 = 1. * torch.ones_like(img)
        K2 = -1. * torch.ones_like(img)
        cur_time = torch.ones((batch_size,), device=device)
        for i, time_step in enumerate(time_steps):
            s = torch.full((batch_size,), time_step, device=device)
            if i == time_steps.shape[0] - 1:
                s = cur_time
            pred = dpm.model(img, cur_time)
            # C, noise = pred.chunk(2, dim=1)
            C, noise = pred[:2]
            # Correct C
            x0 = dpm.pred_x0_from_xt(img, noise, cur_time, K1, K2, C)
            if dpm.clip_x_start:
                x0.clamp_(-1., 1.)
                # C.clamp_(-5/6, 7/6)
                # C.clamp_(-2., 2.)
            C = -1 * x0 - K1 / 3 - K2 / 2
            img = dpm.pred_xtms_from_xt(img, noise, K1, K2, C, cur_time, s)
            # img = self.pred_xtms_from_xt2(img, noise, C, cur_time, s)
            cur_time = cur_time - s
            imgs.append(img)
        show_imgs2(imgs)
        img.clamp_(-1., 1.)

    Css = []
    noisess = []
    for _ in range(100):
        '''
        with torch.no_grad():
            image_size, channels, device = dpm.image_size, dpm.channels, torch.device('cuda:0')
            shape = (batch_size, channels, image_size[0], image_size[1])
            sample_nums = 1000
            time_steps = torch.tensor([1/sample_nums]).repeat(sample_nums)
            # time_steps = torch.tensor([0.25, 0.15, 0.1, 0.1, 0.1, 0.09, 0.075, 0.06, 0.045, 0.03])

            img = torch.randn(shape, device=device)
            imgs = [img]
            # K = -1 * torch.ones_like(img)
            Cs = []
            noises = []
            cur_time = torch.ones((batch_size,), device=device)
            for i, time_step in enumerate(time_steps):
                s = torch.full((batch_size,), time_step, device=device)
                if i == time_steps.shape[0] - 1:
                    s = cur_time
                pred = dpm.model(img, cur_time)
                # C, noise = pred.chunk(2, dim=1)
                C, noise = pred
                if dpm.clip_x_start:
                    C.clamp_(-1., 1.)
                img = pred_xtms_from_xt(img, noise, C, cur_time, s)
                cur_time = cur_time - s
                Cs.append(C.detach().cpu().numpy())
                noises.append(noise.detach().cpu().numpy())
                imgs.append(img)
        Css.append(Cs)
        noisess.append(noises)
        show_imgs(imgs)
        '''
        with torch.no_grad():
            image_size, channels, device = dpm.image_size, dpm.channels, torch.device('cuda:0')
            shape = (batch_size, channels, image_size[0], image_size[1])
            img = torch.randn(shape, device=device)
            sample_nums = 1000
            imgs = [img]
            # K = -1 * torch.ones_like(img)
            Cs = []
            noises = []
            for t in tqdm(reversed(range(0, sample_nums)), desc='sampling loop time step',
                          total=sample_nums):
                self_cond = None
                # img, x_start = dpm.p_sample(img, t, self_cond)
                times = torch.full((img.shape[0],), t, device=device, dtype=torch.long)
                preds = dpm.model_predictions(img, times, self_cond)
                x_start = preds.pred_x_start
                noise = preds.pred_noise
                Cs.append(x_start.detach().cpu().numpy())
                noises.append(noise.detach().cpu().numpy())

                x_start.clamp_(-1., 1.)
                model_mean, posterior_variance, posterior_log_variance = dpm.q_posterior(x_start=x_start, x_t=img, t=times)
                gama = torch.randn_like(x_start) if t > 0 else 0.  # no noise if t == 0
                img = model_mean + (0.5 * posterior_log_variance).exp() * gama
                imgs.append(img)
        Css.append(Cs)
        noisess.append(noises)
        show_imgs(imgs)


    Cs = np.array(Css)
    final = Cs[:, -1]
    Cs = (Cs - final[:,None]) ** 2
    Cs = np.mean(Cs, axis=0)
    # noises = np.mean(np.array(noisess), axis=0)
    y2 = []
    for i in range(len(noisess)):
        tmp = []
        for j in range(len(noisess[i])):
            tmp.append(noisess[i][j].mean())
        y2.append(tmp)
    y2 = np.mean(np.array(y2), axis=0).tolist()

    y3 = []
    for i in range(len(noisess)):
        tmp = []
        for j in range(len(noisess[i])):
            tmp.append(noisess[i][j].std())
        y3.append(tmp)
    y3 = np.mean(np.array(y3), axis=0).tolist()
    # plt.rcParams['font.sans-serif'] = ['Times New Roman']
    fig, ax = plt.subplots()  # 创建图实例
    x = np.linspace(1/len(Cs), 1, len(Cs)).tolist()  # 创建x的取值范围
    y1 = [c.mean().item() for c in Cs]
    with open('./analysis_ddpm.json', 'w') as f:
        json.dump({'time': x,
                   'x0_error': y1,
                   'noise_mean': y2,
                   'noise_std': y3,}, f)
    ax.plot(x, y1, label='x0_error')  # 作y1 = x 图，并标记此线名为linear
    # y2 = [n.mean().item() for n in noises][::-1]
    ax.plot(x, y2, label='noise_mean')  # 作y2 = x^2 图，并标记此线名为quadratic
    # y3 = [n.std().item() for n in noises][::-1]
    ax.plot(x, y3, label='noise_std')  # 作y3 = x^3 图，并标记此线名为cubic
    ax.set_xlabel('denoising time (1-t)')  # 设置x轴名称 x label
    ax.set_ylabel('MSE')  # 设置y轴名称 y label
    # ax.set_title('Simple Plot')  # 设置图名为Simple Plot
    ax.legend()  # 自动检测要在图例中显示的元素，并且显示

    # plt.show()  # 图形可视化
    plt.savefig('./figures/analysis_dpm_100.jpg')
    plt.show()

    ode_sampler = get_ode_sampler(method='RK45', device='cpu')
    x, nfe = ode_sampler(dpm.model, shape)
    x.clamp_(0., 1.)
    pause = 0

def pred_xtms_from_xt(xt, noise, C, t, s):
    time = t.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
    noise = noise / noise.std(dim=[1, 2, 3]).reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
    s = s.reshape(C.shape[0], *((1,) * (len(C.shape) - 1)))
    mean = xt + C * (time-s) - C * time - s / torch.sqrt(time) * noise
    # mean = xt - (C * s * torch.exp(time) + s / torch.sqrt(time) * noise * torch.exp(1-time))
    # mean = xt - (C * s * torch.exp(time) + s / torch.sqrt(time) * noise)
    # mean = xt - (C * s + s / torch.sqrt(time) * noise) * torch.exp(time)
    epsilon = torch.randn_like(mean, device=xt.device)
    sigma = torch.sqrt(s * (time-s) / time)
    xtms = mean + sigma * epsilon
    # xtms = mean
    return xtms

def show_imgs(imgs):
    for ind in range(len(imgs)):
        img = imgs[ind]
        img = unnormalize_to_zero_to_one(img)
        tv.utils.save_image(img, './denoise_imgs/{}.jpg'.format(ind))

def show_imgs2(imgs):
    for ind in range(len(imgs)):
        img = imgs[ind]
        img = unnormalize_to_zero_to_one(img)
        tv.utils.save_image(img, './denoise_imgs/{}.jpg'.format(ind), nrow=4)

def get_ode_sampler(rtol=1e-5, atol=1e-5, denoise=False,
                    method='RK45', eps=1e-3, device='cuda:0', model_type='const'):
  """Probability flow ODE sampler with the black-box ODE solver.

  Args:
    sde: An `sde_lib.SDE` object that represents the forward SDE.
    shape: A sequence of integers. The expected shape of a single sample.
    inverse_scaler: The inverse data normalizer.
    denoise: If `True`, add one-step denoising to final samples.
    rtol: A `float` number. The relative tolerance level of the ODE solver.
    atol: A `float` number. The absolute tolerance level of the ODE solver.
    method: A `str`. The algorithm used for the black-box ODE solver.
      See the documentation of `scipy.integrate.solve_ivp`.
    eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
    device: PyTorch device.

  Returns:
    A sampling function that returns samples and the number of function evaluations during sampling.
  """

  def denoise_update_fn(model, x):
    vec_eps = torch.ones(x.shape[0], device=x.device) * eps
    pred = model(x, vec_eps)
    s = vec_eps.reshape(x.shape[0], *((1,) * (len(x.shape) - 1)))
    C, noise = pred
    mean = x - C * s - s / torch.sqrt(s) * noise
    x = mean
    return x

  def drift_fn(model, x, t, model_type='const'):
    """Get the drift function of the reverse-time SDE."""
    # score_fn = get_score_fn(sde, model, train=False, continuous=True)
    # rsde = sde.reverse(score_fn, probability_flow=True)
    pred = model(x, t)
    if model_type == 'const':
        drift = pred
    if model_type == 'const_sde4':
        C, noise = pred
        drift = -1 * (C + noise / torch.sqrt(t.reshape(noise.shape[0], *((1,) * (len(C.shape) - 1)))))
    elif model_type == 'linear':
        K, C = pred.chunk(2, dim=1)
        drift = K * t + C
    return drift

  def ode_sampler(model, shape):
    """The probability flow ODE sampler with black-box ODE solver.

    Args:
      model: A score model.
      z: If present, generate samples from latent code `z`.
    Returns:
      samples, number of function evaluations.
    """
    with torch.no_grad():
      # Initial sample
      x = torch.randn(*shape)
      def ode_func(t, x):
        x = from_flattened_numpy(x, shape).to(device).type(torch.float32)
        # vec_t = torch.ones(shape[0], device=x.device) * t
        vec_t = torch.ones(shape[0], device=x.device) * t * 1000
        drift = drift_fn(model, x, vec_t, model_type=model_type)
        return to_flattened_numpy(drift)

      # Black-box ODE solver for the probability flow ODE
      solution = integrate.solve_ivp(ode_func, (1, eps), to_flattened_numpy(x),
                                     rtol=rtol, atol=atol, method=method)
      nfe = solution.nfev
      x = torch.tensor(solution.y[:, -1]).reshape(shape).to(device).type(torch.float32)

      # Denoising is equivalent to running one predictor step without adding noise
      if denoise:
        x = denoise_update_fn(model, x)

      # x = inverse_scaler(x)
      x.clamp_(-1., 1.)
      x = unnormalize_to_zero_to_one(x)
      return x, nfe

  return ode_sampler

def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))

def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))


if __name__ == "__main__":
    main()

