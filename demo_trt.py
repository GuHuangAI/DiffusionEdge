import numpy as np
import argparse
import math
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import torchvision as tv
from torch.utils.data import DataLoader
from fvcore.common.config import CfgNode
import time
from denoising_diffusion_pytorch.data import *
import os
import pycuda.driver as cuda
import pycuda.autoinit     # init pycuda.driver
import tensorrt as trt
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="run trt model configure")
    parser.add_argument("--input_dir", help='input directory', type=str, required=True)
    parser.add_argument("--pre_weight", help='path of pretrained weight', type=str, required=True)
    parser.add_argument("--out_dir", help='output directory', type=str, required=True)
    parser.add_argument("--bs", help='batch_size for inference', type=int, default=16)
    parser.add_argument("--crop_size", help='crop size for inference', type=int, default=256)
    # parser.add_argument("")
    args = parser.parse_args()
    # args.cfg = load_conf(args.cfg)
    return args


def main(args):
    torch.manual_seed(42)
    np.random.seed(42)
    dataset = EdgeDatasetTest(
        data_root=args.input_dir,
        image_size=[args.crop_size, args.crop_size],
    )
    dl = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True,
                    num_workers=4)
    # sample_num = model_cfg.sample_num
    # batch_size = sampler_cfg.sample_batch_size
    # batch_num = math.ceil(sample_num // batch_size)
    # save_dir = Path(cfg.save_folder)
    # save_dir.mkdir(exist_ok=True, parents=True)

    sampler = Sampler(data_loader=dl, cfg=args)
    sampler.sample()
    pass


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger()
    assert os.path.exists(engine_file_path)
    print("Reading engine from file {}".format(engine_file_path))
    trt.init_libnvinfer_plugins(None, "")  # 对应op放到了插件库，需加载插件库
    with open(engine_file_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        host_mem: cpu memory
        device_mem: gpu memory
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, binding_dict):
    inputs, outputs, bindings, binding_shapes = [], [], [], []
    for binding in engine:
        # print(binding) # 绑定的输入输出
        binding_instance = binding_dict[binding]
        size = trt.volume(binding_instance.shape)
        # volume 计算可迭代变量的空间，指元素个数
        # size = trt.volume(engine.get_binding_shape(binding)) # 如果采用固定bs的onnx，则采用该句
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # get_binding_dtype  获得binding的数据类型
        # nptype等价于numpy中的dtype，即数据类型
        # allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)  # 创建锁业内存
        device_mem = cuda.mem_alloc(host_mem.nbytes)  # cuda分配空间
        # print(int(device_mem)) # binding在计算图中的缓冲地址
        bindings.append(int(device_mem))
        # append to the appropriate list
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
            binding_shapes.append((engine.get_binding_index(binding), tuple(binding_instance.shape)))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, binding_shapes


def do_inference(context, bindings, inputs, outputs, stream, binding_shapes):
    [context.set_binding_shape(binding_shape[0], binding_shape[1]) for binding_shape in binding_shapes]
    # Transfer data from CPU to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # htod： host to device 将数据由cpu复制到gpu device
    # Run inference.

    time_before = time.time()
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    time_after = time.time()
    # 当创建network时显式指定了batchsize， 则使用execute_async_v2, 否则使用execute_async
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # gpu to cpu
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs], time_before, time_after



class Sampler(object):
    def __init__(
            self,
            data_loader,
            cfg={},
    ):
        super().__init__()
        self.dl = data_loader
        self.cfg = cfg
        self.results_folder = Path(cfg.out_dir)
        self.results_folder.mkdir(exist_ok=True, parents=True)
        # engine_file_path = "model_dym.trt"          # path
        engine_file_path = self.cfg.pre_weight          # path
        engine = load_engine(engine_file_path)
        self.engine = engine
        self.data_batch = self.cfg.bs         # put on cuda data size



    def sample(self):
        with torch.no_grad():
            num = 0
            # time_start = time.time()
            # time_total = 0
            crop_imgs_all = []
            crop_imgs_all_batch = []
            x1s_all = []
            x2s_all = []
            y1s_all = []
            y2s_all = []
            preds_all = []

            # for idx, batch in tqdm(enumerate(self.dl)):
            for idx, batch in enumerate(self.dl):
                for key in batch.keys():
                    if isinstance(batch[key], torch.Tensor):
                        batch[key]
                cond = batch['cond']
                # print(cond)
                # print("cond shape:", cond.shape)
                mask = batch['ori_mask'] if 'ori_mask' in batch else None
                raw_size = batch['raw_size']
                bs = cond.shape[0]

                # size problem
                if cond.shape[2] < self.cfg.crop_size or cond.shape[3] < self.cfg.crop_size:
                    print(f"{batch['img_name']} size is too small!  pass")
                    continue

                time_cur = time.time()
                crop_imgs, x1s, x2s, y1s, y2s, preds = self.slide_crop(cond, crop_size=[self.cfg.crop_size, self.cfg.crop_size],
                                           stride=[self.cfg.crop_size, self.cfg.crop_size], mask=mask, out_channels=1)
                crop_imgs_all.append(crop_imgs)
                crop_imgs_all_batch.append(crop_imgs.shape[0])
                x1s_all.append(x1s)
                x2s_all.append(x2s)
                y1s_all.append(y1s)
                y2s_all.append(y2s)
                preds_all.append(preds)

            crop_seg_logits_all = []
            time_total = 0
            combined_tensor = torch.cat(crop_imgs_all, dim=0)
            img_num = len(crop_imgs_all)

            # backup: "https://aitechtogether.com/python/75070.html"
            # dym: "https://zhuanlan.zhihu.com/p/598735516"
            # initialize trt inference
            init_size = self.cfg.crop_size
            image = torch.randn(self.data_batch, 3, init_size, init_size)
            noise = torch.randn(self.data_batch, 3, int(init_size / 4), int(init_size / 4))
            cur_times = torch.ones(self.data_batch, )
            sampling_timesteps = 1
            step = 1. / sampling_timesteps
            time_steps = torch.tensor([step]).repeat(sampling_timesteps)
            time_steps = time_steps.expand(1, self.data_batch)
            cur_times_expanded = cur_times.unsqueeze(0)
            time_and_step = torch.cat((cur_times_expanded, time_steps), dim=0)

            output = torch.zeros([self.data_batch, 1, init_size, init_size], dtype=torch.int32)
            binding_dict = {
                "noise": noise,
                "time_and_step": time_and_step,
                "input": image,
                "output": output,
            }

            context = self.engine.create_execution_context()
            stream = cuda.Stream()
            inputs, outputs, bindings, binding_shapes = allocate_buffers(self.engine, binding_dict)
            [output], time_before, time_after = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, binding_shapes=binding_shapes)
            # reshape data
            new_size = torch.Size([image.shape[0], 1, image.shape[2], image.shape[3]])
            output_init = output.reshape(new_size)
            print("trt initialization successful，result shape:", output_init.shape)
            print("start inference!")

            for i in range(math.ceil(combined_tensor.shape[0] / self.data_batch)):
                if i == math.ceil(combined_tensor.shape[0] / self.data_batch) - 1:
                    each_tensor = combined_tensor[i * self.data_batch:]
                else:
                    each_tensor = combined_tensor[i*self.data_batch:i*self.data_batch + self.data_batch]
                # print(each_tensor.shape)
                each_batch = each_tensor.shape[0]

                # get noise
                noise_size = int(self.cfg.crop_size / 4)
                noise = torch.randn(each_batch, 3, noise_size, noise_size)

                # get time_and_step
                cur_times = torch.ones(each_batch, )
                sampling_timesteps = 1
                step = 1. / sampling_timesteps
                time_steps = torch.tensor([step]).repeat(sampling_timesteps)
                time_steps = time_steps.expand(1, each_batch)
                # 将维度扩展为 (1, each_batch)
                cur_times_expanded = cur_times.unsqueeze(0)
                # 在第一个维度上进行拼接
                time_and_step = torch.cat((cur_times_expanded, time_steps), dim=0)

                inputs[0].host = noise.reshape(1, -1).detach().cpu().numpy()      # noise
                inputs[1].host = time_and_step.reshape(1, -1).detach().cpu().numpy()  # time_and_step
                inputs[2].host = each_tensor.reshape(1, -1).detach().cpu().numpy()  # input

                # update binding_shapes
                each_binding_shapes = []
                each_binding_shapes.append((0, tuple(noise.shape)))
                each_binding_shapes.append((1, tuple(time_and_step.shape)))
                each_binding_shapes.append((2, tuple(each_tensor.shape)))

                if each_batch != self.data_batch:
                    new_size = each_batch * self.cfg['model']['image_size'][0] * self.cfg['model']['image_size'][0]
                    outputs[0].host = outputs[0].host[:new_size]
                    [output], time_before, time_after = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, binding_shapes=each_binding_shapes)

                [output], time_before, time_after = do_inference(context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream, binding_shapes=each_binding_shapes)

                # reshape data
                new_size = torch.Size([each_tensor.shape[0], 1, each_tensor.shape[2], each_tensor.shape[3]])
                each_crop_seg_logits = output.reshape(new_size)

                print(f"num{i+1} time is {time_after - time_before}")
                time_total += (time_after - time_before)
                each_crop_seg_logits = torch.tensor(each_crop_seg_logits)
                crop_seg_logits_all.append(each_crop_seg_logits)
            print('FPS: ', img_num / time_total)
            crop_seg_logits_tensor = torch.cat(crop_seg_logits_all, dim=0)
            print('inference complete!')
            print('saving images...')
            i = 0
            # for idx, batch in enumerate(self.dl):
            for x1s, x2s, y1s, y2s, preds, batch, img_batch in zip(x1s_all, x2s_all, y1s_all, y2s_all, preds_all, self.dl, crop_imgs_all_batch):
                crop_seg_logits = crop_seg_logits_tensor[i: i + img_batch]
                i += img_batch

                count_mat = preds.clone()

                for crop_seg_logit, x1, x2, y1, y2 in zip(crop_seg_logits, x1s, x2s, y1s, y2s):
                    preds += F.pad(crop_seg_logit, (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)))
                    count_mat[:, :, y1:y2, x1:x2] += 1
                seg_logits = preds / count_mat

                for j, img in enumerate(seg_logits):
                    img_name = batch["img_name"][j]
                    # img[img > 0.5] = 1.           # here
                    num += 1
                    file_name = self.results_folder / img_name
                    tv.utils.save_image(img, str(file_name)[:-4] + ".png")
        print('sampling complete!')

    def slide_crop(self, inputs, crop_size, stride, mask=None, out_channels=1):
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
        out_channels = out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
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

        return crop_imgs, x1s, x2s, y1s, y2s, preds


if __name__ == "__main__":
    args = parse_args()
    main(args)
    pass