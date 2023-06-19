from typing import Any, List, NamedTuple, Optional, Tuple, Union, Callable
import glob
import importlib
import time
import os
import random

import diffusers
import torch
from diffusers import (
    DDPMScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
    DPMSolverSinglestepScheduler,
    LMSDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    KDPM2DiscreteScheduler,
    KDPM2AncestralDiscreteScheduler,
    StableDiffusionPipeline,
)

from pipline_diffusers import (
    CLIP_MODEL_PATH,
    V2_STABLE_DIFFUSION_PATH,
    TOKENIZER_PATH,
    SCHEDULER_TIMESTEPS,
    SCHEDULER_LINEAR_START,
    SCHEDULER_LINEAR_END,
    SCHEDLER_SCHEDULE,
    LATENT_CHANNELS,
    DOWNSAMPLING_FACTOR
)

from pipline_diffusers import (
    replace_unet_modules,
    PipelineLike,
)


from transformers import  CLIPTokenizer, CLIPModel
from PIL import Image

import library.model_util as model_util


class BatchDataBase(NamedTuple):
    # 基础数据
    step: int
    prompt: str
    negative_prompt: str
    seed: int
    clip_skip: int
    init_image: Any
    mask_image: Any
    clip_prompt: str
    guide_image: Any


class BatchDataExt(NamedTuple):
    # 批量数据
    width: int
    height: int
    steps: int
    scale: float
    negative_scale: float
    strength: float
    network_muls: Tuple[float]
    num_sub_prompts: int

class NetWorkData():
    def __init__(self,
                 network_module: str, 
                 network_weight: str, 
                 network_mul: float
                 ) -> None:
        
        self.network_module = network_module
        self.network_weight = network_weight
        self.network_mul = network_mul
        
        # additional argmuments for network (key=value)
        self.network_args = None
        # show metadata of network model
        self.network_show_meta = False
        # "merge network weights to original model
        self.network_merge = False

class Txt2ImgParams():
    def __init__(self,
            sampler: str = "ddim",
            prompt:str = None,
            negative_prompt: str = None,
            steps: int = 30,
            width: int = 512,
            height: int = 512,
            scale: float = 7.5,
            clip_skip: int = 1,
            negative_scale: str = None,
            seed: int = -1,
            batch_size: int = 1,
            clip_prompt: str = None
            ) -> None:
        self.sampler = sampler
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.steps = steps
        self.width = width
        self.height = height
        self.scale = scale
        self.clip_skip = clip_skip
        self.negative_scale = negative_scale
        self.seed = seed
        self.batch_size = batch_size
        self.clip_prompt = clip_prompt
        
        self.networks:tuple[NetWorkData] = []

class BatchData(NamedTuple):
    return_latents: bool
    base: BatchDataBase
    ext: BatchDataExt

# replace randn
class NoiseManager:
    def __init__(self):
        self.sampler_noises = None
        self.sampler_noise_index = 0

    def reset_sampler_noises(self, noises):
        self.sampler_noise_index = 0
        self.sampler_noises = noises

    def randn(self, shape, device=None, dtype=None, layout=None, generator=None):
        # print("replacing", shape, len(self.sampler_noises), self.sampler_noise_index)
        if self.sampler_noises is not None and self.sampler_noise_index < len(self.sampler_noises):
            noise = self.sampler_noises[self.sampler_noise_index]
            if shape != noise.shape:
                noise = None
        else:
            noise = None
        if noise == None:
            print(f"unexpected noise request: {self.sampler_noise_index}, {shape}")
            noise = torch.randn(shape, dtype=dtype, device=device, generator=generator)
        self.sampler_noise_index += 1

        return noise

class TorchRandReplacer:
    def __init__(self, noise_manager):
        self.noise_manager = noise_manager
    def __getattr__(self, item):
        if item == "randn":
            return self.noise_manager.randn
        if hasattr(torch, item):
            return getattr(torch, item)
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, item))


# img2imgの前処理、画像の読み込みなど
def load_images(path):
    if os.path.isfile(path):
        paths = [path]
    else:
        paths = (
            glob.glob(os.path.join(path, "*.png"))
            + glob.glob(os.path.join(path, "*.jpg"))
            + glob.glob(os.path.join(path, "*.jpeg"))
            + glob.glob(os.path.join(path, "*.webp"))
        )
        paths.sort()
    images = []
    for p in paths:
        image = Image.open(p)
        if image.mode != "RGB":
            print(f"convert image to RGB from {image.mode}: {p}")
            image = image.convert("RGB")
        images.append(image)
    return images

def resize_images(imgs, size):
    resized = []
    for img in imgs:
        r_img = img.resize(size, Image.Resampling.LANCZOS)
        if hasattr(img, "filename"):  # filename属性がない場合があるらしい
            r_img.filename = img.filename
        resized.append(r_img)
    return resized

class GenImages():
    def __init__(self):
        # load Stable Diffusion v2.0 model
        self.v2 = False 
        # enable v-parameterization training
        self.v_parameterization = False

        # image to inpaint or to generate from
        self.image_path = None
        # mask in inpainting
        self.mask_path = None
        # img2img strength
        self.strength = None

        self.outdir = None

        # use fp16 / bfloat16 / float32
        self._dtype = torch.float16
        # path to checkpoint of model
        self._ckpt = None
        # path to checkpoint of vae to replace
        self._vae = None

        # directory for caching Tokenizer (for offline training)
        self.tokenizer_cache_dir = None

        # use xformers
        self.xformers = False
        # use xformers by diffusers (Hypernetworks doesn't work) 
        self.diffusers_xformers = False
        
        # set channels last option to model
        self.opt_channels_last = False

        # Embeddings files of Textual Inversion / Textual Inversionのembeddings
        self.textual_inversion_embeddings = None

        # set another guidance scale for negative prompt
        self.negative_scale = None

        # enable CLIP guided SD, scale for guidance (DDIM, PNDM, LMS samplers only)
        self.clip_guidance_scale = 0.0
        # enable CLIP guided SD by image, scale for guidance
        self.clip_image_guidance_scale = 0.0
        # image to CLIP guidance 
        self.guide_image_path = None
        
        self.max_token_length = 0
        # max embeding multiples, max token length is 75 * multiples
        self.max_embeddings_multiples = 0

        # batch size for VAE, < 1.0 for ratio
        self.vae_batch_size = 0

        self._networks = []
        self._network_muls = []
        self._text_encoder = None
        
        self._unet = None
        self._tokenizer = None
        self._clip_model = None
        self._scheduler_num_noises_per_step = 0
        
        self._noise_manager = None
        self._scheduler = None
        self._device = None
        self._pipe = None

        self.img_name_prefix = None # "network"
        self.img_name_type = "default"

    def set_dtype(self, dtype:str=["fp16" ,"bp16"]):
        if dtype == "bp16":
            self._dtype = torch.bfloat16
        else:
            self._dtype = torch.float16

    def set_ckpt(self, ckpt_path:str):
        self._ckpt = ckpt_path

        if not os.path.isfile(ckpt_path):
            files = glob.glob(ckpt_path)
            if len(files) == 1:
                self._ckpt = files[0]
        
        use_stable_diffusion_format = os.path.isfile(self._ckpt)
        if use_stable_diffusion_format:
            print("load StableDiffusion checkpoint")
            self._text_encoder, self._vae, self._unet = model_util.load_models_from_stable_diffusion_checkpoint(self.v2, self._ckpt)
        else:
            print("load Diffusers pretrained models")
            loading_pipe = StableDiffusionPipeline.from_pretrained(self._ckpt, safety_checker=None, torch_dtype=self._dtype)
            self._text_encoder = loading_pipe.text_encoder
            self._vae = loading_pipe.vae
            self._unet = loading_pipe.unet
            self._tokenizer = loading_pipe.tokenizer
            del loading_pipe
        
        if self.clip_guidance_scale  > 0 or self.clip_image_guidance_scale > 0:
            print("prepare clip model")
            self._clip_model = CLIPModel.from_pretrained(CLIP_MODEL_PATH, torch_dtype=self._dtype)

    def _load_tokenizer(self):
        print("loading tokenizer")
        if not os.path.isfile(self._ckpt):
            return
        print("prepare tokenizer")
        original_path = V2_STABLE_DIFFUSION_PATH if self.v2 else TOKENIZER_PATH

        tokenizer: CLIPTokenizer = None
        if self.tokenizer_cache_dir:
            local_tokenizer_path = os.path.join(self.tokenizer_cache_dir, original_path.replace("/", "_"))
            if os.path.exists(local_tokenizer_path):
                print(f"load tokenizer from cache: {local_tokenizer_path}")
                tokenizer = CLIPTokenizer.from_pretrained(local_tokenizer_path)  # same for v1 and v2

        if tokenizer is None:
            if self.v2:
                tokenizer = CLIPTokenizer.from_pretrained(original_path, subfolder="tokenizer")
            else:
                tokenizer = CLIPTokenizer.from_pretrained(original_path)

        if self.max_token_length > 0:
            print(f"update token length: {self.max_token_length}")

        if self.tokenizer_cache_dir and not os.path.exists(local_tokenizer_path):
            print(f"save Tokenizer to cache: {local_tokenizer_path}")
            tokenizer.save_pretrained(local_tokenizer_path)

        self._tokenizer = tokenizer

    def _get_scheduler(self, sampler="ddim"):
        sched_init_args = {}
        self._scheduler_num_noises_per_step = 1

        sampler_dict = {
            "ddim": {
                "scheduler_cls": DDIMScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_ddim
            },
            "ddpm": {
                "scheduler_cls": DDPMScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_ddpm
            },
            "pndm": {
                "scheduler_cls": PNDMScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_pndm
            },
            "lms": {
                "scheduler_cls": LMSDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_lms_discrete
            },
             "k_lms" : {
                "scheduler_cls": LMSDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_lms_discrete
            },
             "euler" : {
                "scheduler_cls": EulerDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_euler_discrete
            },
             "k_euler" : {
                "scheduler_cls": EulerDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_euler_discrete
            },
             "euler_a" : {
                "scheduler_cls": EulerAncestralDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_euler_ancestral_discrete
            },
             "k_euler_a" : {
                "scheduler_cls": EulerAncestralDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_euler_ancestral_discrete
            },
             "dpmsolver" : {
                "scheduler_cls": DPMSolverMultistepScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_dpmsolver_multistep,
            },
             "dpmsolver++" : {
                "scheduler_cls": DPMSolverMultistepScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_dpmsolver_multistep,
            },
             "dpmsingle" : {
                "scheduler_cls": DPMSolverSinglestepScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_dpmsolver_singlestep,
            },
             "heun" : {
                "scheduler_cls": HeunDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_heun_discrete,
            },
             "dpm_2" : {
                "scheduler_cls": KDPM2DiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_k_dpm_2_discrete,
            },
             "k_dpm_2" : {
                "scheduler_cls": KDPM2DiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_k_dpm_2_discrete,
            },
             "dpm_2_a" : {
                "scheduler_cls": HeunDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_heun_discrete,
            },
             "k_dpm_2_a" : {
                "scheduler_cls": KDPM2AncestralDiscreteScheduler,
                "scheduler_module": diffusers.schedulers.scheduling_k_dpm_2_ancestral_discrete,
            }
        }
        
        assert sampler in sampler_dict , f"sampler name {sampler} error." 

        scheduler_cls = sampler_dict[sampler]["scheduler_cls"]
        scheduler_module = sampler_dict[sampler]["scheduler_module"]

        if sampler == "dpmsolver" or sampler == "dpmsolver++":
            sched_init_args["algorithm_type"] = sampler

        if sampler == "dpm_2_a" or sampler == "k_dpm_2_a":
            self._scheduler_num_noises_per_step = 2

        if self.v_parameterization:
            sched_init_args["prediction_type"] = "v_prediction"

        self._noise_manager = NoiseManager()
        
        scheduler_module.torch = TorchRandReplacer(self._noise_manager)

        scheduler = scheduler_cls(
            num_train_timesteps=SCHEDULER_TIMESTEPS,
            beta_start=SCHEDULER_LINEAR_START,
            beta_end=SCHEDULER_LINEAR_END,
            beta_schedule=SCHEDLER_SCHEDULE,
            **sched_init_args,
        )

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is False:
            print("set clip_sample to True")
            scheduler.config.clip_sample = True
        return scheduler
    
    def create_pipline(self):
        # xformers、Hypernetwork 一致
        if not self.diffusers_xformers:
            replace_unet_modules(self._unet, not self.xformers, self.xformers)

        # load tokenizer token分词器
        self._load_tokenizer()

        # 确定 设备类型，使用GPU or CPU
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # "mps"を考量してない

        # custom pipeline
        self._vae.to(self._dtype).to(self._device)
        self._text_encoder.to(self._dtype).to(self._device)
        self._unet.to(self._dtype).to(self._device)
        if self._clip_model is not None:
            self._clip_model.to(self._dtype).to(self._device)
        
        self._pipe = PipelineLike(
            self._device,
            self._vae,
            self._text_encoder,
            self._tokenizer,
            self._unet,
            self._clip_model,
            self.clip_guidance_scale,
            self.clip_image_guidance_scale,
            None,
            0,
            None
        )
        print("pipeline is ready.")

        if self.diffusers_xformers:
            self._pipe.enable_xformers_memory_efficient_attention()

    def load_network(self, networks:tuple[NetWorkData], append_network=True):
        network_list = []

        if len(networks) == 0:
            return
        
        for n in networks:
            print("import network module:", n.network_module)
            imported_module = importlib.import_module(n.network_module)

            net_kwargs = {}
            if n.network_args:
                # TODO escape special chars
                network_args = n.network_args.split(";")
                for net_arg in network_args:
                    key, value = net_arg.split("=")
                    net_kwargs[key] = value

            if not n.network_weight:
                raise ValueError("No weight. Weight is required.")

            print("load network weights from:", n.network_weight)

            if model_util.is_safetensors(n.network_weight) and n.network_show_meta:
                from safetensors.torch import safe_open

                with safe_open(n.network_weight, framework="pt") as f:
                    metadata = f.metadata()
                if metadata is not None:
                    print(f"metadata for: {n.network_weight}: {metadata}")

            network_list.append("%s_%.2f" % (os.path.basename(n.network_weight).split(".")[0], n.network_mul))
            network, weights_sd = imported_module.create_network_from_weights(
                n.network_mul, n.network_weight, self._vae, self._text_encoder, self._unet, for_inference=True, **net_kwargs
            )
                
            if network is None:
                return

            mergiable = hasattr(network, "merge_to")
            if n.network_merge and not mergiable:
                print("network is not mergiable. ignore merge option.")

            if not n.network_merge or not mergiable:
                network.apply_to(self._text_encoder, self._unet)
                info = network.load_state_dict(weights_sd, False)  # network.load_weights
                print(f"weights are loaded: {info}")

                if self.opt_channels_last:
                    network.to(memory_format=torch.channels_last)
                network.to(self._dtype).to(self._device)

                if append_network:
                    self._networks.append(network)
                    self._network_muls.append(n.network_mul)
                else:
                    self._networks=[network]
                    self._network_muls=[n.network_mul]
            else:
                network.merge_to(self._text_encoder, self._unet, weights_sd, self._dtype, self._device)
        
        if self.img_name_type == "network":
            self.img_name_prefix = ",".join(network_list)
        
        if self.opt_channels_last:
            print(f"set optimizing: channels last")
            self._text_encoder.to(memory_format=torch.channels_last)
            self._vae.to(memory_format=torch.channels_last)
            self._unet.to(memory_format=torch.channels_last)
            if self._clip_model is not None:
                self._clip_model.to(memory_format=torch.channels_last)
            for network in self._networks:
                network.to(memory_format=torch.channels_last)

    def _set_embeddings(self):
        if not self.textual_inversion_embeddings:
            return 
        
        token_ids_embeds = []
        for embeds_file in self.textual_inversion_embeddings:
            if model_util.is_safetensors(embeds_file):
                from safetensors.torch import load_file
                data = load_file(embeds_file)
            else:
                data = torch.load(embeds_file, map_location="cpu")

            if "string_to_param" in data:
                data = data["string_to_param"]
            embeds = next(iter(data.values()))

            if type(embeds) != torch.Tensor:
                raise ValueError(f"weight file does not contains Tensor: {embeds_file}")

            num_vectors_per_token = embeds.size()[0]
            token_string = os.path.splitext(os.path.basename(embeds_file))[0]
            token_strings = [token_string] + [f"{token_string}{i+1}" for i in range(num_vectors_per_token - 1)]

            # add new word to tokenizer, count is num_vectors_per_token
            num_added_tokens = self._tokenizer.add_tokens(token_strings)
            if num_added_tokens != num_vectors_per_token:
                return

            token_ids = self._tokenizer.convert_tokens_to_ids(token_strings)
            print(f"Textual Inversion embeddings `{token_string}` loaded. Tokens are added: {token_ids}")
            assert (
                min(token_ids) == token_ids[0] and token_ids[-1] == token_ids[0] + len(token_ids) - 1
            ), f"token ids is not ordered"
            assert len(self._tokenizer) - 1 == token_ids[-1], f"token ids is not end of tokenize: {len(self._tokenizer)}"

            if num_vectors_per_token > 1:
                self._pipe.add_token_replacement(token_ids[0], token_ids)

            token_ids_embeds.append((token_ids, embeds))

        self._text_encoder.resize_token_embeddings(len(self._tokenizer))
        token_embeds = self._text_encoder.get_input_embeddings().weight.data
        for token_ids, embeds in token_ids_embeds:
            for token_id, embed in zip(token_ids, embeds):
                token_embeds[token_id] = embed

    # def _set_pre_image(self, prompt_list):
    #     init_images = None
    #     if self.image_path is not None:
    #         print(f"load image for img2img: {self.image_path}")
    #         init_images = load_images(self.image_path)
    #         assert len(init_images) > 0, f"No image: {self.image_path}"
    #         print(f"loaded {len(init_images)} images for img2img")

    #     mask_images = None
    #     if self.mask_path is not None:
    #         print(f"load mask for inpainting: {self.mask_path}")
    #         mask_images = self.load_images(self.mask_path)
    #         assert len(mask_images) > 0, f"No mask image: {self.image_path}"
    #         print(f"loaded {len(mask_images)} mask images for inpainting")
        
    #     # promptがないとき、画像のPngInfoから取得する
    #     if init_images is not None and len(prompt_list) == 0:
    #         print("get prompts from images' meta data")
    #         for img in init_images:
    #             if "prompt" in img.text:
    #                 prompt = img.text["prompt"]
    #                 if "negative-prompt" in img.text:
    #                     prompt += " --n " + img.text["negative-prompt"]
    #                 prompt_list.append(prompt)

    #         # プロンプトと画像を一致させるため指定回数だけ繰り返す（画像を増幅する）
    #         l = []
    #         for im in init_images:
    #             l.extend([im] * self.images_per_prompt)
    #         init_images = l

    #         if mask_images is not None:
    #             l = []
    #             for im in mask_images:
    #                 l.extend([im] * self.images_per_prompt)
    #             mask_images = l

    #     # 画像サイズにオプション指定があるときはリサイズする
    #     if self.width is not None and self.height is not None:
    #         if init_images is not None:
    #             print(f"resize img2img source images to {self.width}*{self.height}")
    #             init_images = resize_images(init_images, (self.width, self.height))
    #         if mask_images is not None:
    #             print(f"resize img2img mask images to {self.width}*{self.height}")
    #             mask_images = resize_images(mask_images, (self.width, self.height))

    #     if mask_images:
    #         # mask を領域情報として流用する、現在は一回のコマンド呼び出しで1枚だけ対応
    #         print("use mask as region")

    #         size = None
    #         for i, network in enumerate(self._networks):
    #             if i < 3:
    #                 np_mask = np.array(self._mask_images[0])
    #                 np_mask = np_mask[:, :, i]
    #                 size = np_mask.shape
    #             else:
    #                 np_mask = np.full(size, 255, dtype=np.uint8)
    #             mask = torch.from_numpy(np_mask.astype(np.float32) / 255.0)
    #             network.set_region(i, i == len(self._networks) - 1, mask)
    #         mask_images = None

    #     guide_images = None
    #     if self.guide_image_path is not None:
    #         print(f"load image for CLIP/VGG16/ControlNet guidance: {self.guide_image_path}")
    #         guide_images = []
    #         for p in self.guide_image_path:
    #             guide_images.extend(load_images(p))

    #         print(f"loaded {len(guide_images)} guide images for guidance")
    #         if len(guide_images) == 0:
    #             print(f"No guide image, use previous generated image.: {self.image_path}")
    #             guide_images = None

    #     self._init_images = init_images
    #     self._mask_images = mask_images
    #     self._guide_images = guide_images

    def _process_batch(self, batch: List[BatchData]):
        batch_size = len(batch)
        max_embeddings_multiples = 1 if self.max_embeddings_multiples is None else self.max_embeddings_multiples

        # このバッチの情報を取り出す
        (
            return_latents,
            (_, _, _, _, clip_skip, init_image, mask_image, _, guide_image),
            (width, height, steps, scale, negative_scale, strength, _, num_sub_prompts),
        ) = batch[0]
        noise_shape = (LATENT_CHANNELS, height // DOWNSAMPLING_FACTOR, width // DOWNSAMPLING_FACTOR)

        prompts = []
        negative_prompts = []
        start_code = torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
        noises = [
            torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
            for _ in range(steps * self._scheduler_num_noises_per_step)
        ]
        seeds = []
        clip_prompts = []

        if init_image is not None:  # img2img?
            i2i_noises = torch.zeros((batch_size, *noise_shape), device=self._device, dtype=self._dtype)
            init_images = []

            if mask_image is not None:
                mask_images = []
            else:
                mask_images = None
        else:
            i2i_noises = None
            init_images = None
            mask_images = None

        if guide_image is not None:  # CLIP image guided?
            guide_images = []
        else:
            guide_images = None

        # バッチ内の位置に関わらず同じ乱数を使うためにここで乱数を生成しておく。あわせてimage/maskがbatch内で同一かチェックする
        all_images_are_same = True
        all_masks_are_same = True
        all_guide_images_are_same = True
        for i, (_, (_, prompt, negative_prompt, seed, _, init_image, mask_image, clip_prompt, guide_image), _) in enumerate(batch):
            prompts.append(prompt)
            negative_prompts.append(negative_prompt)
            seeds.append(seed)
            clip_prompts.append(clip_prompt)

            if init_image is not None:
                init_images.append(init_image)
                if i > 0 and all_images_are_same:
                    all_images_are_same = init_images[-2] is init_image

            if mask_image is not None:
                mask_images.append(mask_image)
                if i > 0 and all_masks_are_same:
                    all_masks_are_same = mask_images[-2] is mask_image

            if guide_image is not None:
                if type(guide_image) is list:
                    guide_images.extend(guide_image)
                    all_guide_images_are_same = False
                else:
                    guide_images.append(guide_image)
                    if i > 0 and all_guide_images_are_same:
                        all_guide_images_are_same = guide_images[-2] is guide_image

            # make start code
            torch.manual_seed(seed)
            start_code[i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

            # make each noises
            for j in range(steps * self._scheduler_num_noises_per_step):
                noises[j][i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

            if i2i_noises is not None:  # img2img noise
                i2i_noises[i] = torch.randn(noise_shape, device=self._device, dtype=self._dtype)

        self._noise_manager.reset_sampler_noises(noises)

        # すべての画像が同じなら1枚だけpipeに渡すことでpipe側で処理を高速化する
        if init_images is not None and all_images_are_same:
            init_images = init_images[0]
        if mask_images is not None and all_masks_are_same:
            mask_images = mask_images[0]
        if guide_images is not None and all_guide_images_are_same:
            guide_images = guide_images[0]

        # generate
        if self._networks:
            shared = {}
            for n, m in zip(self._networks, self._network_muls):
                n.set_multiplier(m)
                if mask_images:
                    n.set_current_generation(batch_size, num_sub_prompts, width, height, shared)

        images = self._pipe(
            prompts,
            negative_prompts,
            init_images,
            mask_images,
            height,
            width,
            clip_skip,
            steps,
            scale,
            negative_scale,
            strength,
            latents=start_code,
            output_type="pil",
            max_embeddings_multiples=max_embeddings_multiples,
            img2img_noise=i2i_noises,
            vae_batch_size=self.vae_batch_size,
            return_latents=return_latents,
            clip_prompts=clip_prompts,
            clip_guide_images=guide_images,
        )[0]

        # save image
        ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
        images_path = []
        for i, (image, seed) in enumerate(zip(images, seeds)):
            fln = f"im_{ts_str}_{i:03d}_{seed}.png"
            if self.img_name_prefix:
                fln = f"{self.img_name_prefix}_{fln}"
            image_path = os.path.join(self.outdir, fln)
            image.save(image_path)
            images_path.append(image_path)

        return images_path

    def load_vae(self, vae):
         # 单独加载vae
        if vae is not None:
            self._vae = model_util.load_vae(vae, self._dtype)
            print("additional VAE loaded")

    def txt2img(self, param: Txt2ImgParams):
        if self.v_parameterization and not self.v2:
            print("v_parameterization should be with v2")
        if self.v2 and param.clip_skip > 1:
            print("v2 with clip_skip will be unexpected")

        if param.networks:
            # 需要完成后从模型上卸载network
            self.load_network(param.networks, append_network=False)

        # 单独加载simple，当前是在pipline上创建的，需要修改pipline代码。
        # load scheduler 扩散调度器
        self._pipe.set_scheduler(self._get_scheduler(param.sampler))

        # 单独加载embeddings
        self._set_embeddings()

        # 初始化目录
        os.makedirs(self.outdir, exist_ok=True)

        print(f"prompt: {param.prompt}")
        print(f"negative_prompt: {param.negative_prompt}")
        
        first_seed = param.seed
        if first_seed is None or first_seed == -1:
            first_seed = random.randint(0, 0x7FFFFFFF)

        seeds = [first_seed + i for i in range(param.batch_size)]
        global_step = 0
        batch_data = []
        for seed in seeds:  # images_per_prompt数量
            b1 = BatchData(
                False,
                BatchDataBase(global_step, param.prompt, param.negative_prompt, seed, param.clip_skip, None, None, param.clip_prompt, None),
                BatchDataExt(
                    param.width,
                    param.height,
                    param.steps,
                    param.scale,
                    param.negative_scale,
                    0,
                    None,
                    None,
                ),
            )
            batch_data.append(b1)
            global_step += 1

        images_path = self._process_batch(batch_data)
        batch_data.clear()
        return images_path
    
    def gen_once_image(self):
        self.create_pipline()

        self.load_network()

        self.gen_batch_process()