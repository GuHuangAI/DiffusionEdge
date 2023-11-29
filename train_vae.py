import yaml
import argparse
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from denoising_diffusion_pytorch.ema import EMA
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from denoising_diffusion_pytorch.utils import *
import torchvision as tv
from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
from denoising_diffusion_pytorch.data import *
from torch.utils.data import DataLoader
from multiprocessing import cpu_count



def parse_args():
    parser = argparse.ArgumentParser(description="training vae configure")
    parser.add_argument("--cfg", help="experiment configure file name", type=str, required=True)
    # parser.add_argument("")
    args = parser.parse_args()
    args.cfg = load_conf(args.cfg)
    return args


def load_conf(config_file, conf={}):
    with open(config_file) as f:
        exp_conf = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in exp_conf.items():
            conf[k] = v
    return conf

def main(args):
    cfg = args.cfg
    # logger = create_logger(root_dir=cfg['out_path'])
    # writer = SummaryWriter(cfg['out_path'])
    model_cfg = cfg['model']
    model = AutoencoderKL(
        ddconfig=model_cfg['ddconfig'],
        lossconfig=model_cfg['lossconfig'],
        embed_dim=model_cfg['embed_dim'],
        ckpt_path=model_cfg['ckpt_path'],
    )
    data_cfg = cfg["data"]
    if data_cfg['name'] == 'edge':
        dataset = EdgeDataset(
            data_root=data_cfg['img_folder'],
            image_size=model_cfg['ddconfig']['resolution'],
            augment_horizontal_flip=data_cfg['augment_horizontal_flip'],
        )
    else:
        raise NotImplementedError

    dl = DataLoader(dataset, batch_size=data_cfg['batch_size'], shuffle=True, pin_memory=True,
                    num_workers=2)
    train_cfg = cfg['trainer']
    trainer = Trainer(
        model, dl, train_batch_size=data_cfg['batch_size'],
        gradient_accumulate_every=train_cfg['gradient_accumulate_every'],
        train_lr=train_cfg['lr'], train_num_steps=train_cfg['train_num_steps'],
        save_and_sample_every=train_cfg['save_and_sample_every'], results_folder=train_cfg['results_folder'],
        amp=train_cfg['amp'], fp16=train_cfg['fp16'], log_freq= train_cfg['log_freq'], cfg=cfg,
    )
    trainer.train()
    pass


class Trainer(object):
    def __init__(
        self,
        model,
        data_loader,
        train_batch_size = 16,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 100000,
        ema_update_every = 10,
        ema_decay = 0.995,
        save_and_sample_every = 1000,
        num_samples = 25,
        results_folder = './results',
        amp = False,
        fp16 = False,
        split_batches = True,
        log_freq = 10,
        resume_milestone = 0,
        cfg={}
    ):
        super().__init__()
        ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = 'fp16' if fp16 else 'no',
            kwargs_handlers=[ddp_handler],
        )

        self.accelerator.native_amp = amp

        self.model = model

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.log_freq = log_freq

        self.train_num_steps = train_num_steps
        self.image_size = model.encoder.resolution

        # dataset and dataloader

        # self.ds = Dataset(folder, mask_folder, self.image_size, augment_horizontal_flip = augment_horizontal_flip, convert_image_to = convert_image_to)
        # dl = DataLoader(self.ds, batch_size = train_batch_size, shuffle = True, pin_memory = True, num_workers = cpu_count())

        dl = self.accelerator.prepare(data_loader)
        self.dl = cycle(dl)

        # optimizer

        self.opt_ae = torch.optim.AdamW(list(model.encoder.parameters())+
                                  list(model.decoder.parameters())+
                                  list(model.quant_conv.parameters())+
                                  list(model.post_quant_conv.parameters()),
                                  lr=train_lr)
        self.opt_disc = torch.optim.AdamW(model.loss.discriminator.parameters(), lr=train_lr)
        min_lr = cfg['trainer']['min_lr']
        lr_lambda = lambda iter: max((1 - iter / train_num_steps) ** 0.95, min_lr)
        self.lr_scheduler_ae = torch.optim.lr_scheduler.LambdaLR(self.opt_ae, lr_lambda=lr_lambda)
        self.lr_scheduler_disc = torch.optim.lr_scheduler.LambdaLR(self.opt_disc, lr_lambda=lr_lambda)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(model, ema_model=None, beta = ema_decay, update_every = ema_update_every)

            self.results_folder = Path(results_folder)
            self.results_folder.mkdir(exist_ok = True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt_ae, self.opt_disc, self.lr_scheduler_ae, self.lr_scheduler_disc = \
            self.accelerator.prepare(self.model, self.opt_ae, self.opt_disc, self.lr_scheduler_ae,
                                     self.lr_scheduler_disc)
        self.logger = create_logger(root_dir=results_folder)
        self.logger.info(cfg)
        self.writer = SummaryWriter(results_folder)
        self.results_folder = Path(results_folder)
        resume_file = str(self.results_folder / f'model-{resume_milestone}.pt')
        if os.path.isfile(resume_file):
            self.load(resume_milestone)

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt_ae': self.opt_ae.state_dict(),
            'lr_scheduler_ae': self.lr_scheduler_ae.state_dict(),
            'opt_disc': self.opt_disc.state_dict(),
            'lr_scheduler_disc': self.lr_scheduler_disc.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt_ae.load_state_dict(data['opt_ae'])
        self.lr_scheduler_ae.load_state_dict(data['lr_scheduler_ae'])
        self.opt_disc.load_state_dict(data['opt_disc'])
        self.lr_scheduler_disc.load_state_dict(data['lr_scheduler_disc'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data['ema'])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.
                batch = next(self.dl)
                img = batch['image'].to(device)
                for ga_ind in range(self.gradient_accumulate_every):
                    # data = next(self.dl).to(device)
                    # mask = mask.to(device)

                    with self.accelerator.autocast():
                        if isinstance(self.model, nn.parallel.DistributedDataParallel):
                            loss, log_dict  = self.model.module.training_step(img, ga_ind, self.step)
                        else:
                            loss, log_dict = self.model.training_step(img, ga_ind, self.step)

                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        if ga_ind == 0:
                            self.opt_ae.zero_grad()
                            self.opt_disc.zero_grad()
                            self.accelerator.backward(loss)
                            self.opt_ae.step()
                            rec_loss = log_dict["train/rec_loss"]
                            kl_loss = log_dict["train/kl_loss"]
                            d_weight = log_dict["train/d_weight"]
                            disc_factor = log_dict["train/disc_factor"]
                            g_loss = log_dict["train/g_loss"]
                        else:
                            self.opt_disc.zero_grad()
                            self.accelerator.backward(loss)
                            self.opt_disc.step()
                            disc_loss = log_dict["train/disc_loss"]
                            logits_real = log_dict["train/logits_real"]
                            logits_fake = log_dict["train/logits_fake"]

                    if self.step % self.log_freq == 0:
                        log_dict['lr'] = self.opt_ae.param_groups[0]['lr']
                        describtions = dict2str(log_dict)
                        describtions = "[Train Step] {}/{}: ".format(self.step, self.train_num_steps) + describtions
                        if accelerator.is_main_process:
                            pbar.desc = describtions
                            self.logger.info(describtions)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                # pbar.set_description(f'loss: {total_loss:.4f}')
                accelerator.wait_for_everyone()

                self.lr_scheduler_ae.step()
                self.lr_scheduler_disc.step()
                if accelerator.is_main_process:
                    self.writer.add_scalar('Learning_Rate', self.opt_ae.param_groups[0]['lr'], self.step)
                    self.writer.add_scalar('total_loss', total_loss, self.step)
                    self.writer.add_scalar('rec_loss', rec_loss, self.step)
                    self.writer.add_scalar('kl_loss', kl_loss, self.step)
                    self.writer.add_scalar('d_weight', d_weight, self.step)
                    self.writer.add_scalar('disc_factor', disc_factor, self.step)
                    self.writer.add_scalar('g_loss', g_loss, self.step)
                    self.writer.add_scalar('disc_loss', disc_loss, self.step)
                    self.writer.add_scalar('logits_real', logits_real, self.step)
                    self.writer.add_scalar('logits_fake', logits_fake, self.step)

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(device)
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.model.eval()
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            self.save(milestone)
                            # img = self.dl
                            #batches = num_to_groups(self.num_samples, self.batch_size)
                            #all_images_list = list(map(lambda n: self.model.module.validate_img(ns=self.batch_size), batches))
                            if isinstance(self.model, nn.parallel.DistributedDataParallel):
                                all_images = self.model.module.validate_img(img[:2])
                            elif isinstance(self.model, nn.Module):
                                all_images = self.model.validate_img(img[:2])
                            all_images = torch.clamp((all_images + 1.0) / 2.0, min=0.0, max=1.0)

                        nrow = 2 ** math.floor(math.log2(math.sqrt(self.batch_size)))
                        tv.utils.save_image(all_images, str(self.results_folder / f'sample-{milestone}.png'), nrow = nrow)
                        self.model.train()
                pbar.update(1)

        accelerator.print('training complete')


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass