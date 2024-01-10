import numpy as np
import yaml
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from ema_pytorch import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
# from denoising_diffusion_pytorch.transmodel import TransModel
from denoising_diffusion_pytorch.uncond_unet import Unet
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from fvcore.common.config import CfgNode
from scipy import integrate

def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf
def parse_args():
    parser = argparse.ArgumentParser(description="demo configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, default="./configs/default.yaml")
    parser.add_argument("--input_dir", help='input directory', type=str, required=True)
    parser.add_argument("--pre_weight", help='path of pretrained weight', type=str, required=True)
    parser.add_argument("--sampling_timesteps", help='sampling timesteps', type=int, default=1)
    parser.add_argument("--out_dir", help='output directory', type=str, required=True)
    parser.add_argument("--bs", help='batch_size for inference', type=int, default=8)
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def main(args):
    cfg = CfgNode(args.cfg)
    torch.manual_seed(42)
    np.random.seed(42)
    # random.seed(seed)
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg.model
    first_stage_cfg = model_cfg.first_stage
    first_stage_model = AutoencoderKL(
        ddconfig=first_stage_cfg.ddconfig,
        lossconfig=first_stage_cfg.lossconfig,
        embed_dim=first_stage_cfg.embed_dim,
        ckpt_path=first_stage_cfg.ckpt_path,
    )

    if model_cfg.model_name == 'cond_unet':
        from denoising_diffusion_pytorch.mask_cond_unet import Unet
        unet_cfg = model_cfg.unet
        unet = Unet(dim=unet_cfg.dim,
                    channels=unet_cfg.channels,
                    dim_mults=unet_cfg.dim_mults,
                    learned_variance=unet_cfg.get('learned_variance', False),
                    out_mul=unet_cfg.out_mul,
                    cond_in_dim=unet_cfg.cond_in_dim,
                    cond_dim=unet_cfg.cond_dim,
                    cond_dim_mults=unet_cfg.cond_dim_mults,
                    window_sizes1=unet_cfg.window_sizes1,
                    window_sizes2=unet_cfg.window_sizes2,
                    fourier_scale=unet_cfg.fourier_scale,
                    cfg=unet_cfg,
                    )
    else:
        raise NotImplementedError
    if model_cfg.model_type == 'const_sde':
        from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
    else:
        raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
    ldm = LatentDiffusion(
        model=unet,
        auto_encoder=first_stage_model,
        train_sample=model_cfg.train_sample,
        image_size=model_cfg.image_size,
        timesteps=model_cfg.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        loss_type=model_cfg.loss_type,
        objective=model_cfg.objective,
        scale_factor=model_cfg.scale_factor,
        scale_by_std=model_cfg.scale_by_std,
        scale_by_softsign=model_cfg.scale_by_softsign,
        default_scale=model_cfg.get('default_scale', False),
        input_keys=model_cfg.input_keys,
        ckpt_path=model_cfg.ckpt_path,
        ignore_keys=model_cfg.ignore_keys,
        only_model=model_cfg.only_model,
        start_dist=model_cfg.start_dist,
        perceptual_weight=model_cfg.perceptual_weight,
        use_l1=model_cfg.get('use_l1', True),
        cfg=model_cfg,
    )
    # ldm.init_from_ckpt(cfg.sampler.ckpt_path, use_ema=cfg.sampler.get('use_ema', True))

    data_cfg = cfg.data
    if data_cfg['name'] == 'edge':
        dataset = EdgeDatasetTest(
            data_root=args.input_dir,
            image_size=model_cfg.image_size,
        )
        # dataset = torch.utils.data.ConcatDataset([dataset] * 5)
    else:
        raise NotImplementedError
    dl = DataLoader(dataset, batch_size=cfg.sampler.batch_size, shuffle=False, pin_memory=True,
                    num_workers=data_cfg.get('num_workers', 2))
    # for slide sampling, we only support batch size = 1
    sampler_cfg = cfg.sampler
    sampler_cfg.save_folder = args.out_dir
    sampler_cfg.ckpt_path = args.pre_weight
    sampler_cfg.batch_size = args.bs
    sampler = Sampler(
        ldm, dl, batch_size=sampler_cfg.batch_size,
        sample_num=sampler_cfg.sample_num,
        results_folder=sampler_cfg.save_folder, cfg=cfg,
    )
    sampler.sample()


class Sampler(object):
    def __init__(
            self,
            model,
            data_loader,
            sample_num=1000,
            batch_size=16,
            results_folder='./results',
            rk45=False,
            cfg={},
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches=True,
            mixed_precision='no',
            kwargs_handlers=[ddp_handler],
        )
        self.model = model
        self.sample_num = sample_num
        self.rk45 = rk45

        self.batch_size = batch_size
        self.batch_num = math.ceil(sample_num // batch_size)

        self.image_size = model.image_size
        self.cfg = cfg

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = dl
        self.results_folder = Path(results_folder)
        if self.accelerator.is_main_process:
            self.results_folder.mkdir(exist_ok=True, parents=True)

        self.model = self.accelerator.prepare(self.model)
        data = torch.load(cfg.sampler.ckpt_path, map_location=lambda storage, loc: storage)

        model = self.accelerator.unwrap_model(self.model)
        if cfg.sampler.use_ema:
            sd = data['ema']
            new_sd = {}
            for k in sd.keys():
                if k.startswith("ema_model."):
                    new_k = k[10:]  # remove ema_model.
                    new_sd[new_k] = sd[k]
            sd = new_sd
            model.load_state_dict(sd)
        else:
            model.load_state_dict(data['model'])
        if 'scale_factor' in data['model']:
            model.scale_factor = data['model']['scale_factor']

    def sample(self):
        accelerator = self.accelerator
        device = accelerator.device
        batch_num = self.batch_num
        with torch.no_grad():
            self.model.eval()
            psnr = 0.
            num = 0
            for idx, batch in tqdm(enumerate(self.dl)):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key].to(device)
                # image = batch["image"]
                cond = batch['cond']
                raw_w = batch["raw_size"][0].item()      # default batch size = 1
                raw_h = batch["raw_size"][1].item()
                img_name = batch["img_name"][0]

                mask = batch['ori_mask'] if 'ori_mask' in batch else None
                bs = cond.shape[0]
                if self.cfg.sampler.sample_type == 'whole':
                    batch_pred = self.whole_sample(cond, raw_size=(raw_h, raw_w), mask=mask)
                elif self.cfg.sampler.sample_type == 'slide':
                    batch_pred = self.slide_sample(cond, crop_size=self.cfg.sampler.get('crop_size', [320, 320]),
                                                   stride=self.cfg.sampler.stride, mask=mask, bs=self.batch_size)
                else:
                    raise NotImplementedError
                for j, (img, c) in enumerate(zip(batch_pred, cond)):
                    file_name = self.results_folder / img_name
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")
        accelerator.print('sampling complete')

    # ----------------------------------waiting revision------------------------------------
    def slide_sample(self, inputs, crop_size, stride, mask=None, bs=8):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.

        Args:
            inputs (tensor): the tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            batch_img_metas (List[dict]): List of image metainfo where each may
                also contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.

        Returns:
            Tensor: The segmentation results, seg_logits from model of each
                input image.
        """

        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = 1
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out1 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        # aux_out2 = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        crop_imgs = []
        x1s = []
        x2s = []
        y1s = []
        y2s = []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                crop_imgs.append(crop_img)
                x1s.append(x1)
                x2s.append(x2)
                y1s.append(y1)
                y2s.append(y2)
        crop_imgs = torch.cat(crop_imgs, dim=0)
        crop_seg_logits_list = []
        num_windows = crop_imgs.shape[0]
        bs = bs
        length = math.ceil(num_windows / bs)
        for i in range(length):
            if i == length - 1:
                crop_imgs_temp = crop_imgs[bs * i:num_windows, ...]
            else:
                crop_imgs_temp = crop_imgs[bs * i:bs * (i + 1), ...]

            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                crop_seg_logits = self.model.module.sample(batch_size=crop_imgs_temp.shape[0], cond=crop_imgs_temp,
                                                           mask=mask)
            elif isinstance(self.model, nn.Module):
                crop_seg_logits = self.model.sample(batch_size=crop_imgs_temp.shape[0], cond=crop_imgs_temp, mask=mask)
            else:
                raise NotImplementedError

            crop_seg_logits_list.append(crop_seg_logits)
        crop_seg_logits = torch.cat(crop_seg_logits_list, dim=0)
        for crop_seg_logit, x1, x2, y1, y2 in zip(crop_seg_logits, x1s, x2s, y1s, y2s):
            preds += F.pad(crop_seg_logit,
                           (int(x1), int(preds.shape[3] - x2), int(y1),
                            int(preds.shape[2] - y2)))
            count_mat[:, :, y1:y2, x1:x2] += 1

        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        # for h_idx in range(h_grids):
        #     for w_idx in range(w_grids):
        #         y1 = h_idx * h_stride
        #         x1 = w_idx * w_stride
        #         y2 = min(y1 + h_crop, h_img)
        #         x2 = min(x1 + w_crop, w_img)
        #         y1 = max(y2 - h_crop, 0)
        #         x1 = max(x2 - w_crop, 0)
        #         crop_img = inputs[:, :, y1:y2, x1:x2]
        #         # change the image shape to patch shape
        #         # batch_img_metas[0]['img_shape'] = crop_img.shape[2:]
        #         # the output of encode_decode is seg logits tensor map
        #         # with shape [N, C, H, W]
        #         # crop_seg_logit = self.encode_decode(crop_img, batch_img_metas)
        #         if isinstance(self.model, nn.parallel.DistributedDataParallel):
        #             crop_seg_logit = self.model.module.sample(batch_size=1, cond=crop_img, mask=mask)
        #             e1 = e2 = None
        #             aux_out = None
        #         elif isinstance(self.model, nn.Module):
        #             crop_seg_logit = self.model.sample(batch_size=1, cond=crop_img, mask=mask)
        #             e1 = e2 = None
        #             aux_out = None
        #         else:
        #             raise NotImplementedError
        #         preds += F.pad(crop_seg_logit,
        #                        (int(x1), int(preds.shape[3] - x2), int(y1),
        #                         int(preds.shape[2] - y2)))
        #
        #         count_mat[:, :, y1:y2, x1:x2] += 1
        # assert (count_mat == 0).sum() == 0
        # seg_logits = preds / count_mat
        return seg_logits

    def whole_sample(self, inputs, raw_size, mask=None):

        inputs = F.interpolate(inputs, size=(416, 416), mode='bilinear', align_corners=True)

        if isinstance(self.model, nn.parallel.DistributedDataParallel):
            seg_logits = self.model.module.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        elif isinstance(self.model, nn.Module):
            seg_logits = self.model.sample(batch_size=inputs.shape[0], cond=inputs, mask=mask)
        seg_logits = F.interpolate(seg_logits, size=raw_size, mode='bilinear', align_corners=True)
        return seg_logits



if __name__ == "__main__":
    args = parse_args()
    main(args)