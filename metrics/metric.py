from metrics.helpers import get_kwarg, vassert, vprint
from metrics.metric_fid import fid_inputs_to_metric, fid_featuresdict_to_statistics_cached, \
    fid_statistics_to_metric
from metrics.metric_isc import isc_featuresdict_to_metric
from metrics.metric_kid import kid_featuresdict_to_metric
from metrics.metric_ppl import calculate_ppl
from metrics.utils import create_feature_extractor, extract_featuresdict_from_input_id_cached, \
    get_cacheable_input_name
import os
import torch
from denoising_diffusion_pytorch.data import CIFAR10, ImageDataset
from tqdm import tqdm
import numpy as np
import json
from accelerate import Accelerator

def calculate_metrics(**kwargs):
    """
    Calculates metrics for the given inputs. Keyword arguments:

    .. _ISC: https://arxiv.org/pdf/1606.03498.pdf
    .. _FID: https://arxiv.org/pdf/1706.08500.pdf
    .. _KID: https://arxiv.org/pdf/1801.01401.pdf
    .. _PPL: https://arxiv.org/pdf/1812.04948.pdf

    Args:

        input1 (str or torch.utils.data.Dataset or GenerativeModelBase):
            First input, which can be either of the following values:

            - Name of a registered input. See :ref:`registry <Registry>` for the complete list of preregistered
              inputs, and :meth:`register_dataset` for registering a new input. The following options refine the
              behavior wrt dataset location and downloading:
              :paramref:`~calculate_metrics.datasets_root`,
              :paramref:`~calculate_metrics.datasets_download`.
            - Path to a directory with samples. The following options refine the behavior wrt directory
              traversal and samples filtering:
              :paramref:`~calculate_metrics.samples_find_deep`,
              :paramref:`~calculate_metrics.samples_find_ext`, and
              :paramref:`~calculate_metrics.samples_ext_lossy`.
            - Path to a generative model in the :obj:`ONNX<torch:torch.onnx>` or `PTH` (:obj:`JIT<torch:torch.jit>`)
              format. This option also requires the following kwargs:
              :paramref:`~calculate_metrics.input1_model_z_type`,
              :paramref:`~calculate_metrics.input1_model_z_size`, and
              :paramref:`~calculate_metrics.input1_model_num_classes`.
            - Instance of :class:`~torch:torch.utils.data.Dataset` encapsulating a fixed set of samples.
            - Instance of :class:`GenerativeModelBase`, implementing the generative model.

            Default: `None`.

        input2 (str or torch.utils.data.Dataset or GenerativeModelBase):
            Second input, which can be either of the following values:

            - Name of a registered input. See :ref:`registry <Registry>` for the complete list of preregistered
              inputs, and :meth:`register_dataset` for registering a new input. The following options refine the
              behavior wrt dataset location and downloading:
              :paramref:`~calculate_metrics.datasets_root`,
              :paramref:`~calculate_metrics.datasets_download`.
            - Path to a directory with samples. The following options refine the behavior wrt directory
              traversal and samples filtering:
              :paramref:`~calculate_metrics.samples_find_deep`,
              :paramref:`~calculate_metrics.samples_find_ext`, and
              :paramref:`~calculate_metrics.samples_ext_lossy`.
            - Path to a generative model in the :obj:`ONNX<torch:torch.onnx>` or `PTH` (:obj:`JIT<torch:torch.jit>`)
              format. This option also requires the following kwargs:
              :paramref:`~calculate_metrics.input2_model_z_type`,
              :paramref:`~calculate_metrics.input2_model_z_size`, and
              :paramref:`~calculate_metrics.input2_model_num_classes`.
            - Instance of :class:`~torch:torch.utils.data.Dataset` encapsulating a fixed set of samples.
            - Instance of :class:`GenerativeModelBase`, implementing the generative model.

            Default: `None`.

        cuda (bool): Sets executor device to GPU. Default: `True`.

        batch_size (int): Batch size used to process images; the larger the more memory is used on the executor device
            (see :paramref:`~calculate_metrics.cuda`). Default: `64`.

        isc (bool): Calculate ISC_ (Inception Score). Default: `False`.

        fid (bool): Calculate FID_ (Frechet Inception Distance). Default: `False`.

        kid (bool): Calculate KID_ (Kernel Inception Distance). Default: `False`.

        ppl (bool): Calculate PPL_ (Perceptual Path Length). Default: `False`.

        feature_extractor (str): Name of the feature extractor (see :ref:`registry <Registry>`). Default:
            `inception-v3-compat`.

        feature_layer_isc (str): Name of the feature layer to use with ISC metric. Default: `logits_unbiased`.

        feature_layer_fid (str): Name of the feature layer to use with FID metric. Default: `"2048"`.

        feature_layer_kid (str): Name of the feature layer to use with KID metric. Default: `"2048"`.

        feature_extractor_weights_path (str): Path to feature extractor weights (downloaded if `None`). Default: `None`.

        isc_splits (int): Number of splits in ISC. Default: `10`.

        kid_subsets (int): Number of subsets in KID. Default: `100`.

        kid_subset_size (int): Subset size in KID. Default: `1000`.

        kid_degree (int): Degree of polynomial kernel in KID. Default: `3`.

        kid_gamma (float): Polynomial kernel gamma in KID (automatic if `None`). Default: `None`.

        kid_coef0 (float): Polynomial kernel coef0 in KID. Default: `1.0`.

        ppl_epsilon (float): Interpolation step size in PPL. Default: `1e-4`.

        ppl_reduction (str): Reduction type to apply to the per-sample output values. Default: `mean`.

        ppl_sample_similarity (str): Name of the sample similarity to use in PPL metric computation (see :ref:`registry
            <Registry>`). Default: `lpips-vgg16`.

        ppl_sample_similarity_resize (int): Force samples to this size when computing similarity, unless set to `None`.
            Default: `64`.

        ppl_sample_similarity_dtype (str): Check samples are of compatible dtype when computing similarity, unless set
            to `None`. Default: `uint8`.

        ppl_discard_percentile_lower (int): Removes the lower percentile of samples before reduction. Default: `1`.

        ppl_discard_percentile_higher (int): Removes the higher percentile of samples before reduction. Default: `99`.

        ppl_z_interp_mode (str): Noise interpolation mode in PPL (see :ref:`registry <Registry>`). Default: `lerp`.

        samples_shuffle (bool): Perform random samples shuffling before computing splits. Default: `True`.

        samples_find_deep (bool): Find all samples in paths recursively. Default: `False`.

        samples_find_ext (str): List of comma-separated extensions (no blanks) to look for when traversing input path.
            Default: `png,jpg,jpeg`.

        samples_ext_lossy (str): List of comma-separated extensions (no blanks) to warn about lossy compression.
            Default: `jpg,jpeg`.

        datasets_root (str): Path to built-in torchvision datasets root. Default: `$ENV_TORCH_HOME/fidelity_datasets`.

        datasets_download (bool): Download torchvision datasets to :paramref:`~calculate_metrics.dataset_root`.
            Default: `True`.

        cache_root (str): Path to file cache for features and statistics. Default: `$ENV_TORCH_HOME/fidelity_cache`.

        cache (bool): Use file cache for features and statistics. Default: `True`.

        input1_cache_name (str): Assigns a cache entry to input1 (when not a registered input) and forces caching of
            features on it. Default: `None`.

        input1_model_z_type (str): Type of noise, only required when the input is a path to a generator model (see
            :ref:`registry <Registry>`). Default: `normal`.

        input1_model_z_size (int): Dimensionality of noise (only required when the input is a path to a generator
            model). Default: `None`.

        input1_model_num_classes (int): Number of classes for conditional (0 for unconditional) generation (only
            required when the input is a path to a generator model). Default: `0`.

        input1_model_num_samples (int): Number of samples to draw (only required when the input is a generator model).
            This option affects the following metrics: ISC, FID, KID. Default: `None`.

        input2_cache_name (str): Assigns a cache entry to input2 (when not a registered input) and forces caching of
            features on it. Default: `None`.

        input2_model_z_type (str): Type of noise, only required when the input is a path to a generator model (see
            :ref:`registry <Registry>`). Default: `normal`.

        input2_model_z_size (int): Dimensionality of noise (only required when the input is a path to a generator
            model). Default: `None`.

        input2_model_num_classes (int): Number of classes for conditional (0 for unconditional) generation (only
            required when the input is a path to a generator model). Default: `0`.

        input2_model_num_samples (int): Number of samples to draw (only required when the input is a generator model).
            This option affects the following metrics: FID, KID. Default: `None`.

        rng_seed (int): Random numbers generator seed for all operations involving randomness. Default: `2020`.

        save_cpu_ram (bool): Use less CPU RAM at the cost of speed. May not lead to improvement with every metric.
            Default: `False`.

        verbose (bool): Output progress information to STDERR. Default: `True`.

    Returns:

        : Dictionary of metrics with a subset of the following keys:

            - :const:`torch_fidelity.KEY_METRIC_ISC_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_ISC_STD`
            - :const:`torch_fidelity.KEY_METRIC_FID`
            - :const:`torch_fidelity.KEY_METRIC_KID_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_KID_STD`
            - :const:`torch_fidelity.KEY_METRIC_PPL_MEAN`
            - :const:`torch_fidelity.KEY_METRIC_PPL_STD`
            - :const:`torch_fidelity.KEY_METRIC_PPL_RAW`
    """
    config = kwargs.get("cfg")
    # Initialize model
    model_cfg = config.model
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
    elif model_cfg.model_name == 'ncsnpp7':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp7 import create_model
        unet = create_model(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp8':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp8 import create_model
        unet = create_model(unet_cfg)
    elif model_cfg.model_name == 'ncsnpp9':
        unet_cfg = model_cfg.ncsnpp
        from unet_plus.ncsnpp9 import create_model
        unet = create_model(unet_cfg)
    elif model_cfg.model_name == 'restormer':
        unet_cfg = model_cfg.restormer
        from restormer import create_model
        unet = create_model(unet_cfg)
    elif model_cfg.model_name == 'cond_unet3':
        from denoising_diffusion_pytorch.mask_cond_unet3 import Unet
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
                    )
    elif model_cfg.model_name == 'cond_unet4':
        from denoising_diffusion_pytorch.mask_cond_unet4 import Unet
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
    elif model_cfg.model_name == 'cond_unet5':
        from denoising_diffusion_pytorch.mask_cond_unet5 import Unet
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
        from denoising_diffusion_pytorch.mask_cond_unet2 import Unet
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
                    )
    if not model_cfg.ldm:
        if model_cfg.model_type == 'linear':
            from denoising_diffusion_pytorch.etp_dpm_linear import DDPM
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
        elif model_cfg.model_type == 'const_sde3':
            from denoising_diffusion_pytorch.etp_dpm_const_sde3 import DDPM
        elif model_cfg.model_type == 'const_sde4':
            from denoising_diffusion_pytorch.ddm_const_sde import DDPM
        elif model_cfg.model_type == 'linear_sde':
            from denoising_diffusion_pytorch.etp_dpm_linear_sde import DDPM
        elif model_cfg.model_type == '2order_sde':
            from denoising_diffusion_pytorch.etp_dpm_2order_sde import DDPM
        else:
            raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
        generated_model = DDPM(
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
            cfg=model_cfg,
        ).cuda()
    else:
        from denoising_diffusion_pytorch.encoder_decoder import AutoencoderKL
        first_stage_cfg = model_cfg.first_stage
        first_stage_model = AutoencoderKL(
            ddconfig=first_stage_cfg.ddconfig,
            lossconfig=first_stage_cfg.lossconfig,
            embed_dim=first_stage_cfg.embed_dim,
            ckpt_path=first_stage_cfg.ckpt_path,
        )
        if model_cfg.model_type == 'linear':
            from denoising_diffusion_pytorch.etp_ldm_linear import LatentDiffusion
        elif model_cfg.model_type == 'const':
            from denoising_diffusion_pytorch.etp_ldm_const import LatentDiffusion
        elif model_cfg.model_type == 'linear2':
            from denoising_diffusion_pytorch.etp_ldm_linear2 import LatentDiffusion  # without C
        elif model_cfg.model_type == 'exp':
            from denoising_diffusion_pytorch.etp_ldm_exp import LatentDiffusion
        elif model_cfg.model_type == 'exp2':
            from denoising_diffusion_pytorch.etp_ldm_exp2 import LatentDiffusion
        elif model_cfg.model_type == '2order':
            from denoising_diffusion_pytorch.etp_ldm_2order import LatentDiffusion
        elif model_cfg.model_type == '2order_rec':
            from denoising_diffusion_pytorch.etp_ldm_2order_rec import LatentDiffusion
        elif model_cfg.model_type == 'const_sde4':
            from denoising_diffusion_pytorch.ddm_const_sde import LatentDiffusion
        elif model_cfg.model_type == 'linear_sde':
            from denoising_diffusion_pytorch.etp_dpm_linear_sde import LatentDiffusion
        elif model_cfg.model_type == '2order_sde':
            from denoising_diffusion_pytorch.etp_dpm_2order_sde import LatentDiffusion
        else:
            raise NotImplementedError(f'{model_cfg.model_type} is not surportted !')
        generated_model = LatentDiffusion(
            model=unet,
            auto_encoder=first_stage_model,
            train_sample=model_cfg.train_sample,
            image_size=model_cfg.image_size,
            timesteps=model_cfg.timesteps,
            sampling_timesteps=model_cfg.sampling_timesteps,
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
        ).cuda()
    accelerator = Accelerator(
        split_batches=True,
        mixed_precision='no',
    )
    generated_model = accelerator.prepare(generated_model)

    data_cfg = config.data
    if data_cfg.name == 'cifar10':
        dataset = CIFAR10(
            img_folder=data_cfg.img_folder,
            image_size=model_cfg.image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=False
        )
    else:
        dataset = ImageDataset(
            img_folder=data_cfg.img_folder,
            image_size=model_cfg.image_size,
            augment_horizontal_flip=False,
            normalize_to_neg_one_to_one=False
        )
    dataloader_gt = torch.utils.data.DataLoader(dataset,
                                     batch_size=config.eval.batch_size,
                                     drop_last=False,
                                     pin_memory=True,
                                     num_workers=2)

    verbose = get_kwarg('verbose', kwargs)
    input1, input2 = get_kwarg('input1', kwargs), get_kwarg('input2', kwargs)

    have_isc = get_kwarg('isc', kwargs)
    have_fid = get_kwarg('fid', kwargs)
    have_kid = get_kwarg('kid', kwargs)
    have_ppl = get_kwarg('ppl', kwargs)

    need_input1 = have_isc or have_fid or have_kid or have_ppl
    need_input2 = have_fid or have_kid

    vassert(
        have_isc or have_fid or have_kid or have_ppl,
        'At least one of "isc", "fid", "kid", "ppl" metrics must be specified'
    )
    # vassert(input1 is not None or not need_input1, 'First input is required for "isc", "fid", "kid", and "ppl" metrics')
    # vassert(input2 is not None or not need_input2, 'Second input is required for "fid" and "kid" metrics')

    metrics_all = []
    if have_isc or have_fid or have_kid:
        feature_extractor = get_kwarg('feature_extractor', kwargs)
        feature_layer_isc, feature_layer_fid, feature_layer_kid = (None,) * 3
        feature_layers = set()
        if have_isc:
            feature_layer_isc = get_kwarg('feature_layer_isc', kwargs)
            feature_layers.add(feature_layer_isc)
        if have_fid:
            feature_layer_fid = get_kwarg('feature_layer_fid', kwargs)
            feature_layers.add(feature_layer_fid)
        if have_kid:
            feature_layer_kid = get_kwarg('feature_layer_kid', kwargs)
            feature_layers.add(feature_layer_kid)

        feat_extractor = create_feature_extractor(feature_extractor, list(feature_layers), **kwargs)
    for ckpt in range(config.eval.begin_ckpt, config.eval.end_ckpt + 1):
        metrics = {'epoch': ckpt}
        ckpt_path = os.path.join(config.eval.workdir, 'model-{}.pt'.format(ckpt))
        if not os.path.exists(ckpt_path):
            continue
        # generated_model.init_from_ckpt(ckpt_path, use_ema=config.eval.use_ema)
        g_model = accelerator.unwrap_model(generated_model)
        g_model.init_from_ckpt(ckpt_path, use_ema=config.eval.use_ema)
        g_model.eval()
        # isc: input - featuresdict(cached) - metric
        # fid: input - featuresdict(cached) - statistics(cached) - metric
        # kid: input - featuresdict(cached) - metric
        if ckpt == config.eval.begin_ckpt and config.eval.gt_stats is None:
            print('Extracting gt features to {}'.format(os.path.join(config.eval.workdir, 'gt_feature.pth')))
            with torch.no_grad():
                out = None
                for bid, batch in tqdm(enumerate(dataloader_gt)):
                    batch = (batch['image']*255).to(torch.uint8)
                    features = feat_extractor(batch.cuda())
                    featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
                    featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
                    if out is None:
                        out = featuresdict
                    else:
                        out = {k: out[k] + featuresdict[k] for k in out.keys()}
                out = {k: torch.cat(v, dim=0) for k, v in out.items()}
                torch.save(out, os.path.join(config.eval.workdir, 'gt_feature.pth'))

        print('Evaluating ckpt {} Now ...'.format(ckpt))
        num_sampling_rounds = config.eval.num_samples // config.eval.batch_size
        last_batch_size = config.eval.num_samples - num_sampling_rounds * config.eval.batch_size
        with torch.no_grad():
            out = None
            for r in tqdm(range(num_sampling_rounds)):
                if r == num_sampling_rounds - 1 and last_batch_size > 0:
                    bs = last_batch_size
                else:
                    bs = config.eval.batch_size
                # samples = g_model.sample(batch_size=bs)
                if isinstance(generated_model, torch.nn.parallel.DistributedDataParallel):
                    samples = generated_model.module.sample(batch_size=bs)
                elif isinstance(generated_model, torch.nn.Module):
                    samples = generated_model.sample(batch_size=bs)
                else:
                    raise NotImplementedError
                features = feat_extractor((samples.detach()*255).to(torch.uint8))
                featuresdict = feat_extractor.convert_features_tuple_to_dict(features)
                featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
                if out is None:
                    out = featuresdict
                else:
                    out = {k: out[k] + featuresdict[k] for k in out.keys()}
            generated_fea = {k: torch.cat(v, dim=0) for k, v in out.items()}
        featuresdict_1 = generated_fea
        if config.eval.gt_stats is not None:
            featuresdict_2 = torch.load(config.eval.gt_stats)
        else:
            featuresdict_2 = torch.load(os.path.join(config.eval.workdir, 'gt_feature.pth'))

        if have_isc:
            metric_isc = isc_featuresdict_to_metric(featuresdict_1, feature_layer_isc, **kwargs)
            metrics.update(metric_isc)

        if have_fid:
            features1 = featuresdict_1[feature_layer_fid]
            features1 = features1.numpy()
            mu = np.mean(features1, axis=0)
            sigma = np.cov(features1, rowvar=False)
            stats1 = {'mu': mu, 'sigma': sigma}

            features2 = featuresdict_2[feature_layer_fid]
            features2 = features2.numpy()
            mu = np.mean(features2, axis=0)
            sigma = np.cov(features2, rowvar=False)
            stats2 = {'mu': mu, 'sigma': sigma}
            metric_fid = fid_statistics_to_metric(stats1, stats2, get_kwarg('verbose', kwargs))
            metrics.update(metric_fid)

        if have_kid:
            metric_kid = kid_featuresdict_to_metric(featuresdict_1, featuresdict_2, feature_layer_kid, **kwargs)
            metrics.update(metric_kid)

        # if have_ppl:
        #     metric_ppl = calculate_ppl(1, **kwargs)
        #     metrics.update(metric_ppl)

        print('\n'.join((f'{k}: {v:.7g}' for k, v in metrics.items())))
        metrics_all.append(metrics)
    with open(os.path.join(config.eval.workdir, 'metric_results.json'), 'w') as f:
        json.dump(metrics_all, f)
    return metrics_all
