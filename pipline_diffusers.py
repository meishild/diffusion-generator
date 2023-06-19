from typing import List, Optional, Union, Callable
import inspect
from diffusers.utils import deprecate
from diffusers.configuration_utils import FrozenDict
import math
import re

import diffusers
import numpy as np
import torch
import torchvision
from diffusers import (
    AutoencoderKL,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    UNet2DConditionModel,
)
from einops import rearrange
from torch import einsum
from tqdm import tqdm
from torchvision import transforms
from transformers import CLIPTextModel, CLIPTokenizer, CLIPModel
import PIL
from PIL import Image

from networks.lora import LoRANetwork
import tools.original_control_net as original_control_net
from tools.original_control_net import ControlNetInfo

# Tokenizer: checkpointから読み込むのではなくあらかじめ提供されているものを使う
TOKENIZER_PATH = "openai/clip-vit-large-patch14"
V2_STABLE_DIFFUSION_PATH = "stabilityai/stable-diffusion-2"  # ここからtokenizerだけ使う

DEFAULT_TOKEN_LENGTH = 75

# scheduler:
SCHEDULER_LINEAR_START = 0.00085
SCHEDULER_LINEAR_END = 0.0120
SCHEDULER_TIMESTEPS = 1000
SCHEDLER_SCHEDULE = "scaled_linear"

# その他の設定
LATENT_CHANNELS = 4
DOWNSAMPLING_FACTOR = 8

# CLIP_ID_L14_336 = "openai/clip-vit-large-patch14-336"

# CLIP guided SD関連
CLIP_MODEL_PATH = "laion/CLIP-ViT-B-32-laion2B-s34B-b79K"
FEATURE_EXTRACTOR_SIZE = (224, 224)
FEATURE_EXTRACTOR_IMAGE_MEAN = [0.48145466, 0.4578275, 0.40821073]
FEATURE_EXTRACTOR_IMAGE_STD = [0.26862954, 0.26130258, 0.27577711]

VGG16_IMAGE_MEAN = [0.485, 0.456, 0.406]
VGG16_IMAGE_STD = [0.229, 0.224, 0.225]
VGG16_INPUT_RESIZE_DIV = 4

# CLIP特徴量の取得時にcutoutを使うか：使う場合にはソースを書き換えてください
NUM_CUTOUTS = 4
USE_CUTOUTS = False

# region モジュール入れ替え部
"""
高速化のためのモジュール入れ替え
"""

# FlashAttentionを使うCrossAttention
# based on https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/memory_efficient_attention_pytorch/flash_attention.py
# LICENSE MIT https://github.com/lucidrains/memory-efficient-attention-pytorch/blob/main/LICENSE

# constants

EPSILON = 1e-6

# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards

# https://arxiv.org/abs/2205.14135


class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """Algorithm 2 in the paper"""

        device = q.device
        dtype = q.dtype
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), dtype=dtype, device=device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, dtype=dtype, device=device)

        scale = q.shape[-1] ** -0.5

        if not exists(mask):
            mask = (None,) * math.ceil(q.shape[-2] / q_bucket_size)
        else:
            mask = rearrange(mask, "b n -> b 1 1 n")
            mask = mask.split(q_bucket_size, dim=-1)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            mask,
            all_row_sums.split(q_bucket_size, dim=-2),
            all_row_maxes.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if exists(row_mask):
                    attn_weights.masked_fill_(~row_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim=-1, keepdims=True)
                attn_weights -= block_row_maxes
                exp_weights = torch.exp(attn_weights)

                if exists(row_mask):
                    exp_weights.masked_fill_(~row_mask, 0.0)

                block_row_sums = exp_weights.sum(dim=-1, keepdims=True).clamp(min=EPSILON)

                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_values = einsum("... i j, ... j d -> ... i d", exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)
                exp_block_row_max_diff = torch.exp(block_row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + exp_block_row_max_diff * block_row_sums

                oc.mul_((row_sums / new_row_sums) * exp_row_max_diff).add_((exp_block_row_max_diff / new_row_sums) * exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, all_row_sums, all_row_maxes)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """Algorithm 4 in the paper"""

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, l, m = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim=-2),
            o.split(q_bucket_size, dim=-2),
            do.split(q_bucket_size, dim=-2),
            mask,
            l.split(q_bucket_size, dim=-2),
            m.split(q_bucket_size, dim=-2),
            dq.split(q_bucket_size, dim=-2),
        )

        for ind, (qc, oc, doc, row_mask, lc, mc, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim=-2),
                v.split(k_bucket_size, dim=-2),
                dk.split(k_bucket_size, dim=-2),
                dv.split(k_bucket_size, dim=-2),
            )

            for k_ind, (kc, vc, dkc, dvc) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum("... i d, ... j d -> ... i j", qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), dtype=torch.bool, device=device).triu(
                        q_start_index - k_start_index + 1
                    )
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                exp_attn_weights = torch.exp(attn_weights - mc)

                if exists(row_mask):
                    exp_attn_weights.masked_fill_(~row_mask, 0.0)

                p = exp_attn_weights / lc

                dv_chunk = einsum("... i j, ... i d -> ... j d", p, doc)
                dp = einsum("... i d, ... j d -> ... i j", doc, vc)

                D = (doc * oc).sum(dim=-1, keepdims=True)
                ds = p * scale * (dp - D)

                dq_chunk = einsum("... i j, ... j d -> ... i d", ds, kc)
                dk_chunk = einsum("... i j, ... i d -> ... j d", ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None


def replace_unet_modules(unet: diffusers.models.unet_2d_condition.UNet2DConditionModel, mem_eff_attn, xformers):
    if mem_eff_attn:
        replace_unet_cross_attn_to_memory_efficient()
    elif xformers:
        replace_unet_cross_attn_to_xformers()


def replace_unet_cross_attn_to_memory_efficient():
    print("Replace CrossAttention.forward to use NAI style Hypernetwork and FlashAttention")
    flash_func = FlashAttentionFunction

    def forward_flash_attn(self, x, context=None, mask=None):
        q_bucket_size = 512
        k_bucket_size = 1024

        h = self.heads
        q = self.to_q(x)

        context = context if context is not None else x
        context = context.to(x.dtype)

        if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k = self.to_k(context_k)
        v = self.to_v(context_v)
        del context, x

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), (q, k, v))

        out = flash_func.apply(q, k, v, mask, False, q_bucket_size, k_bucket_size)

        out = rearrange(out, "b h n d -> b n (h d)")

        # diffusers 0.7.0~
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_flash_attn


def replace_unet_cross_attn_to_xformers():
    print("Replace CrossAttention.forward to use NAI style Hypernetwork and xformers")
    try:
        import xformers.ops
    except ImportError:
        raise ImportError("No xformers / xformersがインストールされていないようです")

    def forward_xformers(self, x, context=None, mask=None):
        h = self.heads
        q_in = self.to_q(x)

        context = default(context, x)
        context = context.to(x.dtype)

        if hasattr(self, "hypernetwork") and self.hypernetwork is not None:
            context_k, context_v = self.hypernetwork.forward(x, context)
            context_k = context_k.to(x.dtype)
            context_v = context_v.to(x.dtype)
        else:
            context_k = context
            context_v = context

        k_in = self.to_k(context_k)
        v_in = self.to_v(context_v)

        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b n h d", h=h), (q_in, k_in, v_in))
        del q_in, k_in, v_in

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None)  # 最適なのを選んでくれる

        out = rearrange(out, "b n h d -> b n (h d)", h=h)

        # diffusers 0.7.0~
        out = self.to_out[0](out)
        out = self.to_out[1](out)
        return out

    diffusers.models.attention.CrossAttention.forward = forward_xformers

class PipelineLike:
    r"""
    Pipeline for text-to-image generation using Stable Diffusion without tokens length limit, and support parsing
    weighting in prompt.
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)
    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`StableDiffusionSafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    def __init__(
        self,
        device,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        clip_model: CLIPModel,
        clip_guidance_scale: float,
        clip_image_guidance_scale: float,
        vgg16_model: torchvision.models.VGG,
        vgg16_guidance_scale: float,
        vgg16_layer_no: int,
        # safety_checker: StableDiffusionSafetyChecker,
        # feature_extractor: CLIPFeatureExtractor,
    ):
        super().__init__()
        self.device = device
        self.vae = vae
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.unet = unet
        self.safety_checker = None

        # Textual Inversion
        self.token_replacements = {}

        # XTI
        self.token_replacements_XTI = {}

        # CLIP guidance
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_image_guidance_scale = clip_image_guidance_scale
        self.clip_model = clip_model
        self.normalize = transforms.Normalize(mean=FEATURE_EXTRACTOR_IMAGE_MEAN, std=FEATURE_EXTRACTOR_IMAGE_STD)
        self.make_cutouts = MakeCutouts(FEATURE_EXTRACTOR_SIZE)

        # VGG16 guidance
        self.vgg16_guidance_scale = vgg16_guidance_scale
        if self.vgg16_guidance_scale > 0.0:
            return_layers = {f"{vgg16_layer_no}": "feat"}
            self.vgg16_feat_model = torchvision.models._utils.IntermediateLayerGetter(
                vgg16_model.features, return_layers=return_layers
            )
            self.vgg16_normalize = transforms.Normalize(mean=VGG16_IMAGE_MEAN, std=VGG16_IMAGE_STD)

        # ControlNet
        self.control_nets: List[ControlNetInfo] = []

    def set_scheduler(self, scheduler):
        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)
        self.scheduler = scheduler

    # Textual Inversion
    def add_token_replacement(self, target_token_id, rep_token_ids):
        self.token_replacements[target_token_id] = rep_token_ids

    def replace_token(self, tokens, layer=None):
        new_tokens = []
        for token in tokens:
            if token in self.token_replacements:
                replacer_ = self.token_replacements[token]
                if layer:
                    replacer = []
                for r in replacer_:
                    if r in self.token_replacements_XTI:
                        replacer.append(self.token_replacements_XTI[r][layer])
                    else:
                        replacer = replacer_
                new_tokens.extend(replacer)
            else:
                new_tokens.append(token)
        return new_tokens

    def add_token_replacement_XTI(self, target_token_id, rep_token_ids):
        self.token_replacements_XTI[target_token_id] = rep_token_ids

    def set_control_nets(self, ctrl_nets):
        self.control_nets = ctrl_nets

    # region xformersとか使う部分：独自に書き換えるので関係なし

    def enable_xformers_memory_efficient_attention(self):
        r"""
        Enable memory efficient attention as implemented in xformers.
        When this option is enabled, you should observe lower GPU memory usage and a potential speed up at inference
        time. Speed up at training time is not guaranteed.
        Warning: When Memory Efficient Attention and Sliced attention are both enabled, the Memory Efficient Attention
        is used.
        """
        self.unet.set_use_memory_efficient_attention_xformers(True)

    def disable_xformers_memory_efficient_attention(self):
        r"""
        Disable memory efficient attention as implemented in xformers.
        """
        self.unet.set_use_memory_efficient_attention_xformers(False)

    def enable_attention_slicing(self, slice_size: Optional[Union[str, int]] = "auto"):
        r"""
        Enable sliced attention computation.
        When this option is enabled, the attention module will split the input tensor in slices, to compute attention
        in several steps. This is useful to save some memory in exchange for a small speed decrease.
        Args:
            slice_size (`str` or `int`, *optional*, defaults to `"auto"`):
                When `"auto"`, halves the input to the attention heads, so attention will be computed in two steps. If
                a number is provided, uses as many slices as `attention_head_dim // slice_size`. In this case,
                `attention_head_dim` must be a multiple of `slice_size`.
        """
        if slice_size == "auto":
            # half the attention head size is usually a good trade-off between
            # speed and memory
            slice_size = self.unet.config.attention_head_dim // 2
        self.unet.set_attention_slice(slice_size)

    def disable_attention_slicing(self):
        r"""
        Disable sliced attention computation. If `enable_attention_slicing` was previously invoked, this method will go
        back to computing attention in one step.
        """
        # set slice_size = `None` to disable `attention slicing`
        self.enable_attention_slicing(None)

    def enable_sequential_cpu_offload(self):
        r"""
        Offloads all models to CPU using accelerate, significantly reducing memory usage. When called, unet,
        text_encoder, vae and safety checker have their state dicts saved to CPU and then are moved to a
        `torch.device('meta') and loaded to GPU only when their specific submodule has its `forward` method called.
        """
        # accelerateが必要になるのでとりあえず省略
        raise NotImplementedError("cpu_offload is omitted.")
        # if is_accelerate_available():
        #   from accelerate import cpu_offload
        # else:
        #   raise ImportError("Please install accelerate via `pip install accelerate`")

        # device = self.device

        # for cpu_offloaded_model in [self.unet, self.text_encoder, self.vae, self.safety_checker]:
        #   if cpu_offloaded_model is not None:
        #     cpu_offload(cpu_offloaded_model, device)

    # endregion

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        init_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        mask_image: Union[torch.FloatTensor, PIL.Image.Image, List[PIL.Image.Image]] = None,
        height: int = 512,
        width: int = 512,
        clip_skip: int = 1,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_scale: float = None,
        strength: float = 0.8,
        # num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        vae_batch_size: float = None,
        return_latents: bool = False,
        # return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        is_cancelled_callback: Optional[Callable[[], bool]] = None,
        callback_steps: Optional[int] = 1,
        img2img_noise=None,
        clip_prompts=None,
        clip_guide_images=None,
        networks: Optional[List[LoRANetwork]] = None,
        **kwargs,
    ):
        r"""
        Function invoked when calling the pipeline for generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            is_cancelled_callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. If the function returns
                `True`, the inference will be cancelled.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            `None` if cancelled by `is_cancelled_callback`,
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        num_images_per_prompt = 1  # fixed

        if isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        reginonal_network = " AND " in prompt[0]

        vae_batch_size = (
            batch_size
            if vae_batch_size is None
            else (int(vae_batch_size) if vae_batch_size >= 1 else max(1, int(batch_size * vae_batch_size)))
        )

        if strength < 0 or strength > 1:
            raise ValueError(f"The value of strength should in [0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type" f" {type(callback_steps)}."
            )

        # get prompt text embeddings

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        if not do_classifier_free_guidance and negative_scale is not None:
            print(f"negative_scale is ignored if guidance scalle <= 1.0")
            negative_scale = None

        # get unconditional embeddings for classifier free guidance
        if negative_prompt is None:
            negative_prompt = [""] * batch_size
        elif isinstance(negative_prompt, str):
            negative_prompt = [negative_prompt] * batch_size
        if batch_size != len(negative_prompt):
            raise ValueError(
                f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                " the batch size of `prompt`."
            )

        if not self.token_replacements_XTI:
            text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,
                uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip,
                **kwargs,
            )

        if negative_scale is not None:
            _, real_uncond_embeddings, _ = get_weighted_text_embeddings(
                pipe=self,
                prompt=prompt,  # こちらのトークン長に合わせてuncondを作るので75トークン超で必須
                uncond_prompt=[""] * batch_size,
                max_embeddings_multiples=max_embeddings_multiples,
                clip_skip=clip_skip,
                **kwargs,
            )

        if self.token_replacements_XTI:
            text_embeddings_concat = []
            for layer in [
                "IN01",
                "IN02",
                "IN04",
                "IN05",
                "IN07",
                "IN08",
                "MID",
                "OUT03",
                "OUT04",
                "OUT05",
                "OUT06",
                "OUT07",
                "OUT08",
                "OUT09",
                "OUT10",
                "OUT11",
            ]:
                text_embeddings, uncond_embeddings, prompt_tokens = get_weighted_text_embeddings(
                    pipe=self,
                    prompt=prompt,
                    uncond_prompt=negative_prompt if do_classifier_free_guidance else None,
                    max_embeddings_multiples=max_embeddings_multiples,
                    clip_skip=self.clip_skip,
                    layer=layer,
                    **kwargs,
                )
                if do_classifier_free_guidance:
                    if negative_scale is None:
                        text_embeddings_concat.append(torch.cat([uncond_embeddings, text_embeddings]))
                    else:
                        text_embeddings_concat.append(torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings]))
                text_embeddings = torch.stack(text_embeddings_concat)
        else:
            if do_classifier_free_guidance:
                if negative_scale is None:
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
                else:
                    text_embeddings = torch.cat([uncond_embeddings, text_embeddings, real_uncond_embeddings])

        # CLIP guidanceで使用するembeddingsを取得する
        if self.clip_guidance_scale > 0:
            clip_text_input = prompt_tokens
            if clip_text_input.shape[1] > self.tokenizer.model_max_length:
                # TODO 75文字を超えたら警告を出す？
                print("trim text input", clip_text_input.shape)
                clip_text_input = torch.cat(
                    [clip_text_input[:, : self.tokenizer.model_max_length - 1], clip_text_input[:, -1].unsqueeze(1)], dim=1
                )
                print("trimmed", clip_text_input.shape)

            for i, clip_prompt in enumerate(clip_prompts):
                if clip_prompt is not None:  # clip_promptがあれば上書きする
                    clip_text_input[i] = self.tokenizer(
                        clip_prompt,
                        padding="max_length",
                        max_length=self.tokenizer.model_max_length,
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids.to(self.device)

            text_embeddings_clip = self.clip_model.get_text_features(clip_text_input)
            text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)  # prompt複数件でもOK

        if (
            self.clip_image_guidance_scale > 0
            or self.vgg16_guidance_scale > 0
            and clip_guide_images is not None
            or self.control_nets
        ):
            if isinstance(clip_guide_images, PIL.Image.Image):
                clip_guide_images = [clip_guide_images]

            if self.clip_image_guidance_scale > 0:
                clip_guide_images = [preprocess_guide_image(im) for im in clip_guide_images]
                clip_guide_images = torch.cat(clip_guide_images, dim=0)

                clip_guide_images = self.normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)
                image_embeddings_clip = self.clip_model.get_image_features(clip_guide_images)
                image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
                if len(image_embeddings_clip) == 1:
                    image_embeddings_clip = image_embeddings_clip.repeat((batch_size, 1, 1, 1))
            elif self.vgg16_guidance_scale > 0:
                size = (width // VGG16_INPUT_RESIZE_DIV, height // VGG16_INPUT_RESIZE_DIV)  # とりあえず1/4に（小さいか?）
                clip_guide_images = [preprocess_vgg16_guide_image(im, size) for im in clip_guide_images]
                clip_guide_images = torch.cat(clip_guide_images, dim=0)

                clip_guide_images = self.vgg16_normalize(clip_guide_images).to(self.device).to(text_embeddings.dtype)
                image_embeddings_vgg16 = self.vgg16_feat_model(clip_guide_images)["feat"]
                if len(image_embeddings_vgg16) == 1:
                    image_embeddings_vgg16 = image_embeddings_vgg16.repeat((batch_size, 1, 1, 1))
            else:
                # ControlNetのhintにguide imageを流用する
                # 前処理はControlNet側で行う
                pass

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps, self.device)

        latents_dtype = text_embeddings.dtype
        init_latents_orig = None
        mask = None

        if init_image is None:
            # get the initial random noise unless the user supplied it

            # Unlike in other pipelines, latents need to be generated in the target device
            # for 1-to-1 results reproducibility with the CompVis implementation.
            # However this currently doesn't work in `mps`.
            latents_shape = (
                batch_size * num_images_per_prompt,
                self.unet.in_channels,
                height // 8,
                width // 8,
            )

            if latents is None:
                if self.device.type == "mps":
                    # randn does not exist on mps
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device="cpu",
                        dtype=latents_dtype,
                    ).to(self.device)
                else:
                    latents = torch.randn(
                        latents_shape,
                        generator=generator,
                        device=self.device,
                        dtype=latents_dtype,
                    )
            else:
                if latents.shape != latents_shape:
                    raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {latents_shape}")
                latents = latents.to(self.device)

            timesteps = self.scheduler.timesteps.to(self.device)

            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        else:
            # image to tensor
            if isinstance(init_image, PIL.Image.Image):
                init_image = [init_image]
            if isinstance(init_image[0], PIL.Image.Image):
                init_image = [preprocess_image(im) for im in init_image]
                init_image = torch.cat(init_image)
            if isinstance(init_image, list):
                init_image = torch.stack(init_image)

            # mask image to tensor
            if mask_image is not None:
                if isinstance(mask_image, PIL.Image.Image):
                    mask_image = [mask_image]
                if isinstance(mask_image[0], PIL.Image.Image):
                    mask_image = torch.cat([preprocess_mask(im) for im in mask_image])  # H*W, 0 for repaint

            # encode the init image into latents and scale the latents
            init_image = init_image.to(device=self.device, dtype=latents_dtype)
            if init_image.size()[-2:] == (height // 8, width // 8):
                init_latents = init_image
            else:
                if vae_batch_size >= batch_size:
                    init_latent_dist = self.vae.encode(init_image).latent_dist
                    init_latents = init_latent_dist.sample(generator=generator)
                else:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    init_latents = []
                    for i in tqdm(range(0, batch_size, vae_batch_size)):
                        init_latent_dist = self.vae.encode(
                            init_image[i : i + vae_batch_size] if vae_batch_size > 1 else init_image[i].unsqueeze(0)
                        ).latent_dist
                        init_latents.append(init_latent_dist.sample(generator=generator))
                    init_latents = torch.cat(init_latents)

                init_latents = 0.18215 * init_latents

            if len(init_latents) == 1:
                init_latents = init_latents.repeat((batch_size, 1, 1, 1))
            init_latents_orig = init_latents

            # preprocess mask
            if mask_image is not None:
                mask = mask_image.to(device=self.device, dtype=latents_dtype)
                if len(mask) == 1:
                    mask = mask.repeat((batch_size, 1, 1, 1))

                # check sizes
                if not mask.shape == init_latents.shape:
                    raise ValueError("The mask and init_image should be the same size!")

            # get the original timestep using init_timestep
            offset = self.scheduler.config.get("steps_offset", 0)
            init_timestep = int(num_inference_steps * strength) + offset
            init_timestep = min(init_timestep, num_inference_steps)

            timesteps = self.scheduler.timesteps[-init_timestep]
            timesteps = torch.tensor([timesteps] * batch_size * num_images_per_prompt, device=self.device)

            # add noise to latents using the timesteps
            latents = self.scheduler.add_noise(init_latents, img2img_noise, timesteps)

            t_start = max(num_inference_steps - init_timestep + offset, 0)
            timesteps = self.scheduler.timesteps[t_start:].to(self.device)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        num_latent_input = (3 if negative_scale is not None else 2) if do_classifier_free_guidance else 1

        if self.control_nets:
            guided_hints = original_control_net.get_guided_hints(self.control_nets, num_latent_input, batch_size, clip_guide_images)

        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = latents.repeat((num_latent_input, 1, 1, 1))
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            if self.control_nets:
                if reginonal_network:
                    num_sub_and_neg_prompts = len(text_embeddings) // batch_size
                    text_emb_last = text_embeddings[num_sub_and_neg_prompts - 2 :: num_sub_and_neg_prompts]  # last subprompt
                else:
                    text_emb_last = text_embeddings
                noise_pred = original_control_net.call_unet_and_control_net(
                    i,
                    num_latent_input,
                    self.unet,
                    self.control_nets,
                    guided_hints,
                    i / len(timesteps),
                    latent_model_input,
                    t,
                    text_emb_last,
                ).sample
            else:
                noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            if do_classifier_free_guidance:
                if negative_scale is None:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(num_latent_input)  # uncond by negative prompt
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
                else:
                    noise_pred_negative, noise_pred_text, noise_pred_uncond = noise_pred.chunk(
                        num_latent_input
                    )  # uncond is real uncond
                    noise_pred = (
                        noise_pred_uncond
                        + guidance_scale * (noise_pred_text - noise_pred_uncond)
                        - negative_scale * (noise_pred_negative - noise_pred_uncond)
                    )

            # perform clip guidance
            if self.clip_guidance_scale > 0 or self.clip_image_guidance_scale > 0 or self.vgg16_guidance_scale > 0:
                text_embeddings_for_guidance = (
                    text_embeddings.chunk(num_latent_input)[1] if do_classifier_free_guidance else text_embeddings
                )

                if self.clip_guidance_scale > 0:
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        text_embeddings_clip,
                        self.clip_guidance_scale,
                        NUM_CUTOUTS,
                        USE_CUTOUTS,
                    )
                if self.clip_image_guidance_scale > 0 and clip_guide_images is not None:
                    noise_pred, latents = self.cond_fn(
                        latents,
                        t,
                        i,
                        text_embeddings_for_guidance,
                        noise_pred,
                        image_embeddings_clip,
                        self.clip_image_guidance_scale,
                        NUM_CUTOUTS,
                        USE_CUTOUTS,
                    )
                if self.vgg16_guidance_scale > 0 and clip_guide_images is not None:
                    noise_pred, latents = self.cond_fn_vgg16(
                        latents, t, i, text_embeddings_for_guidance, noise_pred, image_embeddings_vgg16, self.vgg16_guidance_scale
                    )

            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

            if mask is not None:
                # masking
                init_latents_proper = self.scheduler.add_noise(init_latents_orig, img2img_noise, torch.tensor([t]))
                latents = (init_latents_proper * mask) + (latents * (1 - mask))

            # call the callback, if provided
            if i % callback_steps == 0:
                if callback is not None:
                    callback(i, t, latents)
                if is_cancelled_callback is not None and is_cancelled_callback():
                    return None

        if return_latents:
            return (latents, False)

        latents = 1 / 0.18215 * latents
        if vae_batch_size >= batch_size:
            image = self.vae.decode(latents).sample
        else:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            images = []
            for i in tqdm(range(0, batch_size, vae_batch_size)):
                images.append(
                    self.vae.decode(latents[i : i + vae_batch_size] if vae_batch_size > 1 else latents[i].unsqueeze(0)).sample
                )
            image = torch.cat(images)

        image = (image / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16
        image = image.cpu().permute(0, 2, 3, 1).float().numpy()

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(self.numpy_to_pil(image), return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=image,
                clip_input=safety_checker_input.pixel_values.to(text_embeddings.dtype),
            )
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            # image = self.numpy_to_pil(image)
            image = (image * 255).round().astype("uint8")
            image = [Image.fromarray(im) for im in image]

        # if not return_dict:
        return (image, has_nsfw_concept)

        # return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)

    def text2img(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for text-to-image generation.
        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            height (`int`, *optional*, defaults to 512):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to 512):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    def img2img(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for image-to-image generation.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to transform the reference `init_image`. Must be between 0 and 1.
                `init_image` will be used as a starting point, adding more noise to it the larger the `strength`. The
                number of denoising steps depends on the amount of noise initially added. When `strength` is 1, added
                noise will be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `init_image`.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter will be modulated by `strength`.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    def inpaint(
        self,
        init_image: Union[torch.FloatTensor, PIL.Image.Image],
        mask_image: Union[torch.FloatTensor, PIL.Image.Image],
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        strength: float = 0.8,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[torch.Generator] = None,
        max_embeddings_multiples: Optional[int] = 3,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        **kwargs,
    ):
        r"""
        Function for inpaint.
        Args:
            init_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. This is the image whose masked region will be inpainted.
            mask_image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, to mask `init_image`. White pixels in the mask will be
                replaced by noise and therefore repainted, while black pixels will be preserved. If `mask_image` is a
                PIL image, it will be converted to a single channel (luminance) before use. If it's a tensor, it should
                contain one color channel (L) instead of 3, so the expected shape would be `(B, H, W, 1)`.
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            strength (`float`, *optional*, defaults to 0.8):
                Conceptually, indicates how much to inpaint the masked area. Must be between 0 and 1. When `strength`
                is 1, the denoising process will be run on the masked area for the full number of iterations specified
                in `num_inference_steps`. `init_image` will be used as a reference for the masked area, adding more
                noise to that region the larger the `strength`. If `strength` is 0, no inpainting will occur.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The reference number of denoising steps. More denoising steps usually lead to a higher quality image at
                the expense of slower inference. This parameter will be modulated by `strength`, as explained above.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            max_embeddings_multiples (`int`, *optional*, defaults to `3`):
                The max multiple length of prompt embeddings compared to the max output length of text encoder.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        return self.__call__(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=init_image,
            mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            num_images_per_prompt=num_images_per_prompt,
            eta=eta,
            generator=generator,
            max_embeddings_multiples=max_embeddings_multiples,
            output_type=output_type,
            return_dict=return_dict,
            callback=callback,
            callback_steps=callback_steps,
            **kwargs,
        )

    # CLIP guidance StableDiffusion
    # copy from https://github.com/huggingface/diffusers/blob/main/examples/community/clip_guided_stable_diffusion.py

    # バッチを分解して1件ずつ処理する
    def cond_fn(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        guide_embeddings_clip,
        clip_guidance_scale,
        num_cutouts,
        use_cutouts=True,
    ):
        if len(latents) == 1:
            return self.cond_fn1(
                latents,
                timestep,
                index,
                text_embeddings,
                noise_pred_original,
                guide_embeddings_clip,
                clip_guidance_scale,
                num_cutouts,
                use_cutouts,
            )

        noise_pred = []
        cond_latents = []
        for i in range(len(latents)):
            lat1 = latents[i].unsqueeze(0)
            tem1 = text_embeddings[i].unsqueeze(0)
            npo1 = noise_pred_original[i].unsqueeze(0)
            gem1 = guide_embeddings_clip[i].unsqueeze(0)
            npr1, cla1 = self.cond_fn1(lat1, timestep, index, tem1, npo1, gem1, clip_guidance_scale, num_cutouts, use_cutouts)
            noise_pred.append(npr1)
            cond_latents.append(cla1)

        noise_pred = torch.cat(noise_pred)
        cond_latents = torch.cat(cond_latents)
        return noise_pred, cond_latents

    @torch.enable_grad()
    def cond_fn1(
        self,
        latents,
        timestep,
        index,
        text_embeddings,
        noise_pred_original,
        guide_embeddings_clip,
        clip_guidance_scale,
        num_cutouts,
        use_cutouts=True,
    ):
        latents = latents.detach().requires_grad_()

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = latents

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / 0.18215 * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)

        if use_cutouts:
            image = self.make_cutouts(image, num_cutouts)
        else:
            image = transforms.Resize(FEATURE_EXTRACTOR_SIZE)(image)
        image = self.normalize(image).to(latents.dtype)

        image_embeddings_clip = self.clip_model.get_image_features(image)
        image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)

        if use_cutouts:
            dists = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip)
            dists = dists.view([num_cutouts, sample.shape[0], -1])
            loss = dists.sum(2).mean(0).sum() * clip_guidance_scale
        else:
            # バッチサイズが複数だと正しく動くかわからない
            loss = spherical_dist_loss(image_embeddings_clip, guide_embeddings_clip).mean() * clip_guidance_scale

        grads = -torch.autograd.grad(loss, latents)[0]

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents

    # バッチを分解して一件ずつ処理する
    def cond_fn_vgg16(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale):
        if len(latents) == 1:
            return self.cond_fn_vgg16_b1(
                latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale
            )

        noise_pred = []
        cond_latents = []
        for i in range(len(latents)):
            lat1 = latents[i].unsqueeze(0)
            tem1 = text_embeddings[i].unsqueeze(0)
            npo1 = noise_pred_original[i].unsqueeze(0)
            gem1 = guide_embeddings[i].unsqueeze(0)
            npr1, cla1 = self.cond_fn_vgg16_b1(lat1, timestep, index, tem1, npo1, gem1, guidance_scale)
            noise_pred.append(npr1)
            cond_latents.append(cla1)

        noise_pred = torch.cat(noise_pred)
        cond_latents = torch.cat(cond_latents)
        return noise_pred, cond_latents

    # 1件だけ処理する
    @torch.enable_grad()
    def cond_fn_vgg16_b1(self, latents, timestep, index, text_embeddings, noise_pred_original, guide_embeddings, guidance_scale):
        latents = latents.detach().requires_grad_()

        if isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            # the model input needs to be scaled to match the continuous ODE formulation in K-LMS
            latent_model_input = latents / ((sigma**2 + 1) ** 0.5)
        else:
            latent_model_input = latents

        # predict the noise residual
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if isinstance(self.scheduler, (PNDMScheduler, DDIMScheduler)):
            alpha_prod_t = self.scheduler.alphas_cumprod[timestep]
            beta_prod_t = 1 - alpha_prod_t
            # compute predicted original sample from predicted noise also called
            # "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
            pred_original_sample = (latents - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

            fac = torch.sqrt(beta_prod_t)
            sample = pred_original_sample * (fac) + latents * (1 - fac)
        elif isinstance(self.scheduler, LMSDiscreteScheduler):
            sigma = self.scheduler.sigmas[index]
            sample = latents - sigma * noise_pred
        else:
            raise ValueError(f"scheduler type {type(self.scheduler)} not supported")

        sample = 1 / 0.18215 * sample
        image = self.vae.decode(sample).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        image = transforms.Resize((image.shape[-2] // VGG16_INPUT_RESIZE_DIV, image.shape[-1] // VGG16_INPUT_RESIZE_DIV))(image)
        image = self.vgg16_normalize(image).to(latents.dtype)

        image_embeddings = self.vgg16_feat_model(image)["feat"]

        # バッチサイズが複数だと正しく動くかわからない
        loss = ((image_embeddings - guide_embeddings) ** 2).mean() * guidance_scale  # MSE style transferでコンテンツの損失はMSEなので

        grads = -torch.autograd.grad(loss, latents)[0]
        if isinstance(self.scheduler, LMSDiscreteScheduler):
            latents = latents.detach() + grads * (sigma**2)
            noise_pred = noise_pred_original
        else:
            noise_pred = noise_pred_original - torch.sqrt(beta_prod_t) * grads
        return noise_pred, latents


class MakeCutouts(torch.nn.Module):
    def __init__(self, cut_size, cut_power=1.0):
        super().__init__()

        self.cut_size = cut_size
        self.cut_power = cut_power

    def forward(self, pixel_values, num_cutouts):
        sideY, sideX = pixel_values.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(num_cutouts):
            size = int(torch.rand([]) ** self.cut_power * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = pixel_values[:, :, offsety : offsety + size, offsetx : offsetx + size]
            cutouts.append(torch.nn.functional.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = torch.nn.functional.normalize(x, dim=-1)
    y = torch.nn.functional.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


re_attention = re.compile(
    r"""
\\\(|
\\\)|
\\\[|
\\]|
\\\\|
\\|
\(|
\[|
:([+-]?[.\d]+)\)|
\)|
]|
[^\\()\[\]:]+|
:
""",
    re.X,
)


def parse_prompt_attention(text):
    """
    Parses a string with attention tokens and returns a list of pairs: text and its associated weight.
    Accepted tokens are:
      (abc) - increases attention to abc by a multiplier of 1.1
      (abc:3.12) - increases attention to abc by a multiplier of 3.12
      [abc] - decreases attention to abc by a multiplier of 1.1
      \( - literal character '('
      \[ - literal character '['
      \) - literal character ')'
      \] - literal character ']'
      \\ - literal character '\'
      anything else - just text
    >>> parse_prompt_attention('normal text')
    [['normal text', 1.0]]
    >>> parse_prompt_attention('an (important) word')
    [['an ', 1.0], ['important', 1.1], [' word', 1.0]]
    >>> parse_prompt_attention('(unbalanced')
    [['unbalanced', 1.1]]
    >>> parse_prompt_attention('\(literal\]')
    [['(literal]', 1.0]]
    >>> parse_prompt_attention('(unnecessary)(parens)')
    [['unnecessaryparens', 1.1]]
    >>> parse_prompt_attention('a (((house:1.3)) [on] a (hill:0.5), sun, (((sky))).')
    [['a ', 1.0],
     ['house', 1.5730000000000004],
     [' ', 1.1],
     ['on', 1.0],
     [' a ', 1.1],
     ['hill', 0.55],
     [', sun, ', 1.1],
     ['sky', 1.4641000000000006],
     ['.', 1.1]]
    """

    res = []
    round_brackets = []
    square_brackets = []

    round_bracket_multiplier = 1.1
    square_bracket_multiplier = 1 / 1.1

    def multiply_range(start_position, multiplier):
        for p in range(start_position, len(res)):
            res[p][1] *= multiplier

    for m in re_attention.finditer(text):
        text = m.group(0)
        weight = m.group(1)

        if text.startswith("\\"):
            res.append([text[1:], 1.0])
        elif text == "(":
            round_brackets.append(len(res))
        elif text == "[":
            square_brackets.append(len(res))
        elif weight is not None and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), float(weight))
        elif text == ")" and len(round_brackets) > 0:
            multiply_range(round_brackets.pop(), round_bracket_multiplier)
        elif text == "]" and len(square_brackets) > 0:
            multiply_range(square_brackets.pop(), square_bracket_multiplier)
        else:
            res.append([text, 1.0])

    for pos in round_brackets:
        multiply_range(pos, round_bracket_multiplier)

    for pos in square_brackets:
        multiply_range(pos, square_bracket_multiplier)

    if len(res) == 0:
        res = [["", 1.0]]

    # merge runs of identical weights
    i = 0
    while i + 1 < len(res):
        if res[i][1] == res[i + 1][1]:
            res[i][0] += res[i + 1][0]
            res.pop(i + 1)
        else:
            i += 1

    return res


def get_prompts_with_weights(pipe: PipelineLike, prompt: List[str], max_length: int, layer=None):
    r"""
    Tokenize a list of prompts and return its tokens with weights of each token.
    No padding, starting or ending token is included.
    """
    tokens = []
    weights = []
    truncated = False
    for text in prompt:
        texts_and_weights = parse_prompt_attention(text)
        text_token = []
        text_weight = []
        for word, weight in texts_and_weights:
            # tokenize and discard the starting and the ending token
            token = pipe.tokenizer(word).input_ids[1:-1]

            token = pipe.replace_token(token, layer=layer)

            text_token += token
            # copy the weight by length of token
            text_weight += [weight] * len(token)
            # stop if the text is too long (longer than truncation limit)
            if len(text_token) > max_length:
                truncated = True
                break
        # truncate
        if len(text_token) > max_length:
            truncated = True
            text_token = text_token[:max_length]
            text_weight = text_weight[:max_length]
        tokens.append(text_token)
        weights.append(text_weight)
    if truncated:
        print("warning: Prompt was truncated. Try to shorten the prompt or increase max_embeddings_multiples")
    return tokens, weights


def pad_tokens_and_weights(tokens, weights, max_length, bos, eos, pad, no_boseos_middle=True, chunk_length=77):
    r"""
    Pad the tokens (with starting and ending tokens) and weights (with 1.0) to max_length.
    """
    max_embeddings_multiples = (max_length - 2) // (chunk_length - 2)
    weights_length = max_length if no_boseos_middle else max_embeddings_multiples * chunk_length
    for i in range(len(tokens)):
        tokens[i] = [bos] + tokens[i] + [eos] + [pad] * (max_length - 2 - len(tokens[i]))
        if no_boseos_middle:
            weights[i] = [1.0] + weights[i] + [1.0] * (max_length - 1 - len(weights[i]))
        else:
            w = []
            if len(weights[i]) == 0:
                w = [1.0] * weights_length
            else:
                for j in range(max_embeddings_multiples):
                    w.append(1.0)  # weight for starting token in this chunk
                    w += weights[i][j * (chunk_length - 2) : min(len(weights[i]), (j + 1) * (chunk_length - 2))]
                    w.append(1.0)  # weight for ending token in this chunk
                w += [1.0] * (weights_length - len(w))
            weights[i] = w[:]

    return tokens, weights


def get_unweighted_text_embeddings(
    pipe: PipelineLike,
    text_input: torch.Tensor,
    chunk_length: int,
    clip_skip: int,
    eos: int,
    pad: int,
    no_boseos_middle: Optional[bool] = True,
):
    """
    When the length of tokens is a multiple of the capacity of the text encoder,
    it should be split into chunks and sent to the text encoder individually.
    """
    max_embeddings_multiples = (text_input.shape[1] - 2) // (chunk_length - 2)
    if max_embeddings_multiples > 1:
        text_embeddings = []
        for i in range(max_embeddings_multiples):
            # extract the i-th chunk
            text_input_chunk = text_input[:, i * (chunk_length - 2) : (i + 1) * (chunk_length - 2) + 2].clone()

            # cover the head and the tail by the starting and the ending tokens
            text_input_chunk[:, 0] = text_input[0, 0]
            if pad == eos:  # v1
                text_input_chunk[:, -1] = text_input[0, -1]
            else:  # v2
                for j in range(len(text_input_chunk)):
                    if text_input_chunk[j, -1] != eos and text_input_chunk[j, -1] != pad:  # 最後に普通の文字がある
                        text_input_chunk[j, -1] = eos
                    if text_input_chunk[j, 1] == pad:  # BOSだけであとはPAD
                        text_input_chunk[j, 1] = eos

            if clip_skip is None or clip_skip == 1:
                text_embedding = pipe.text_encoder(text_input_chunk)[0]
            else:
                enc_out = pipe.text_encoder(text_input_chunk, output_hidden_states=True, return_dict=True)
                text_embedding = enc_out["hidden_states"][-clip_skip]
                text_embedding = pipe.text_encoder.text_model.final_layer_norm(text_embedding)

            if no_boseos_middle:
                if i == 0:
                    # discard the ending token
                    text_embedding = text_embedding[:, :-1]
                elif i == max_embeddings_multiples - 1:
                    # discard the starting token
                    text_embedding = text_embedding[:, 1:]
                else:
                    # discard both starting and ending tokens
                    text_embedding = text_embedding[:, 1:-1]

            text_embeddings.append(text_embedding)
        text_embeddings = torch.concat(text_embeddings, axis=1)
    else:
        if clip_skip is None or clip_skip == 1:
            text_embeddings = pipe.text_encoder(text_input)[0]
        else:
            enc_out = pipe.text_encoder(text_input, output_hidden_states=True, return_dict=True)
            text_embeddings = enc_out["hidden_states"][-clip_skip]
            text_embeddings = pipe.text_encoder.text_model.final_layer_norm(text_embeddings)
    return text_embeddings


def get_weighted_text_embeddings(
    pipe: PipelineLike,
    prompt: Union[str, List[str]],
    uncond_prompt: Optional[Union[str, List[str]]] = None,
    max_embeddings_multiples: Optional[int] = 1,
    no_boseos_middle: Optional[bool] = False,
    skip_parsing: Optional[bool] = False,
    skip_weighting: Optional[bool] = False,
    clip_skip=None,
    layer=None,
    **kwargs,
):
    r"""
    Prompts can be assigned with local weights using brackets. For example,
    prompt 'A (very beautiful) masterpiece' highlights the words 'very beautiful',
    and the embedding tokens corresponding to the words get multiplied by a constant, 1.1.
    Also, to regularize of the embedding, the weighted embedding would be scaled to preserve the original mean.
    Args:
        pipe (`DiffusionPipeline`):
            Pipe to provide access to the tokenizer and the text encoder.
        prompt (`str` or `List[str]`):
            The prompt or prompts to guide the image generation.
        uncond_prompt (`str` or `List[str]`):
            The unconditional prompt or prompts for guide the image generation. If unconditional prompt
            is provided, the embeddings of prompt and uncond_prompt are concatenated.
        max_embeddings_multiples (`int`, *optional*, defaults to `1`):
            The max multiple length of prompt embeddings compared to the max output length of text encoder.
        no_boseos_middle (`bool`, *optional*, defaults to `False`):
            If the length of text token is multiples of the capacity of text encoder, whether reserve the starting and
            ending token in each of the chunk in the middle.
        skip_parsing (`bool`, *optional*, defaults to `False`):
            Skip the parsing of brackets.
        skip_weighting (`bool`, *optional*, defaults to `False`):
            Skip the weighting. When the parsing is skipped, it is forced True.
    """
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2
    if isinstance(prompt, str):
        prompt = [prompt]

    # split the prompts with "AND". each prompt must have the same number of splits
    new_prompts = []
    for p in prompt:
        new_prompts.extend(p.split(" AND "))
    prompt = new_prompts

    if not skip_parsing:
        prompt_tokens, prompt_weights = get_prompts_with_weights(pipe, prompt, max_length - 2, layer=layer)
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens, uncond_weights = get_prompts_with_weights(pipe, uncond_prompt, max_length - 2, layer=layer)
    else:
        prompt_tokens = [token[1:-1] for token in pipe.tokenizer(prompt, max_length=max_length, truncation=True).input_ids]
        prompt_weights = [[1.0] * len(token) for token in prompt_tokens]
        if uncond_prompt is not None:
            if isinstance(uncond_prompt, str):
                uncond_prompt = [uncond_prompt]
            uncond_tokens = [
                token[1:-1] for token in pipe.tokenizer(uncond_prompt, max_length=max_length, truncation=True).input_ids
            ]
            uncond_weights = [[1.0] * len(token) for token in uncond_tokens]

    # round up the longest length of tokens to a multiple of (model_max_length - 2)
    max_length = max([len(token) for token in prompt_tokens])
    if uncond_prompt is not None:
        max_length = max(max_length, max([len(token) for token in uncond_tokens]))

    max_embeddings_multiples = min(
        max_embeddings_multiples,
        (max_length - 1) // (pipe.tokenizer.model_max_length - 2) + 1,
    )
    max_embeddings_multiples = max(1, max_embeddings_multiples)
    max_length = (pipe.tokenizer.model_max_length - 2) * max_embeddings_multiples + 2

    # pad the length of tokens and weights
    bos = pipe.tokenizer.bos_token_id
    eos = pipe.tokenizer.eos_token_id
    pad = pipe.tokenizer.pad_token_id
    prompt_tokens, prompt_weights = pad_tokens_and_weights(
        prompt_tokens,
        prompt_weights,
        max_length,
        bos,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
        chunk_length=pipe.tokenizer.model_max_length,
    )
    prompt_tokens = torch.tensor(prompt_tokens, dtype=torch.long, device=pipe.device)
    if uncond_prompt is not None:
        uncond_tokens, uncond_weights = pad_tokens_and_weights(
            uncond_tokens,
            uncond_weights,
            max_length,
            bos,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
            chunk_length=pipe.tokenizer.model_max_length,
        )
        uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=pipe.device)

    # get the embeddings
    text_embeddings = get_unweighted_text_embeddings(
        pipe,
        prompt_tokens,
        pipe.tokenizer.model_max_length,
        clip_skip,
        eos,
        pad,
        no_boseos_middle=no_boseos_middle,
    )
    prompt_weights = torch.tensor(prompt_weights, dtype=text_embeddings.dtype, device=pipe.device)
    if uncond_prompt is not None:
        uncond_embeddings = get_unweighted_text_embeddings(
            pipe,
            uncond_tokens,
            pipe.tokenizer.model_max_length,
            clip_skip,
            eos,
            pad,
            no_boseos_middle=no_boseos_middle,
        )
        uncond_weights = torch.tensor(uncond_weights, dtype=uncond_embeddings.dtype, device=pipe.device)

    # assign weights to the prompts and normalize in the sense of mean
    # TODO: should we normalize by chunk or in a whole (current implementation)?
    # →全体でいいんじゃないかな
    if (not skip_parsing) and (not skip_weighting):
        previous_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= prompt_weights.unsqueeze(-1)
        current_mean = text_embeddings.float().mean(axis=[-2, -1]).to(text_embeddings.dtype)
        text_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)
        if uncond_prompt is not None:
            previous_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= uncond_weights.unsqueeze(-1)
            current_mean = uncond_embeddings.float().mean(axis=[-2, -1]).to(uncond_embeddings.dtype)
            uncond_embeddings *= (previous_mean / current_mean).unsqueeze(-1).unsqueeze(-1)

    if uncond_prompt is not None:
        return text_embeddings, uncond_embeddings, prompt_tokens
    return text_embeddings, None, prompt_tokens


def preprocess_guide_image(image):
    image = image.resize(FEATURE_EXTRACTOR_SIZE, resample=Image.NEAREST)  # cond_fnと合わせる
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # nchw
    image = torch.from_numpy(image)
    return image  # 0 to 1


# VGG16の入力は任意サイズでよいので入力画像を適宜リサイズする
def preprocess_vgg16_guide_image(image, size):
    image = image.resize(size, resample=Image.NEAREST)  # cond_fnと合わせる
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)  # nchw
    image = torch.from_numpy(image)
    return image  # 0 to 1


def preprocess_image(image):
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask):
    mask = mask.convert("L")
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=PIL.Image.BILINEAR)  # LANCZOS)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask