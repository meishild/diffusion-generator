import os
import torch
import random

from typing import List, Optional, Union, Callable

from diffusers import (
    EulerDiscreteScheduler,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline
)
from transformers import CLIPTextModel
import pipline_diffusers

project_path = os.path.abspath(".")

# pip install safetensors omegaconf
pipe = StableDiffusionXLPipeline.from_single_file(
    "E:\\ai-stable-diffsuion\\SDXL0.9\\13G version\\sd_xl_base_0.9.safetensors",
    torch_dtype=torch.float16).to("cuda")

text_encoder = pipe.text_encoder
# 需要支持单独加载vae
vae = pipe.vae
unet = pipe.unet
tokenizer = pipe.tokenizer
scheduler = pipe.scheduler

# https://zhuanlan.zhihu.com/p/639330929
# pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)
pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

token_embeds = text_encoder.get_input_embeddings().weight.data

# https://github.com/huggingface/diffusers/issues/3212
# clip_skip???
# num_hidden_layers=10
clip_skip=2

# text_encoder = CLIPTextModel.from_pretrained(
#     "openai/clip-vit-large-patch14",
#     # subfolder="text_encoder",
#     num_hidden_layers=13-clip_skip,
#     torch_dtype=torch.float16,
# ).to("cuda")
# pipe.text_encoder = text_encoder

pipe.load_textual_inversion(os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors"), token="EasyNegative")
# pipe.unet.enable_xformers_memory_efficient_attention()

# pipe.load_lora_weights(os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"))

pipe.text_encoder = CLIPTextModel._from_config(pipe.text_encoder.config, torch_dtype=torch.float16).to("cuda")

# 当前不支持加权
prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, medium breasts,beautiful detailed eyes"
# negative_prompt = "EasyNegative"
negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans,extra fingers,fewer fingers,((watermark:2)),((white letters:1))"

seed = random.randint(0, 0x7FFFFFFF)
width = 1024
height = 1024

generator = torch.Generator("cuda").manual_seed(seed)

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7,
    negative_prompt=negative_prompt,
    height=height,
    width=width,
    generator=generator,
).images[0]

outdir = os.path.join(project_path, "outputs")
image.save(os.path.join(outdir, "test.png"))