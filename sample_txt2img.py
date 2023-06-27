import os
import torch

from diffusers import StableDiffusionPipeline

project_path = os.path.abspath(".")
cache_dir = [os.path.join(project_path, ".cache\huggingface\transformers")]

# pip install safetensors omegaconf
pipe = StableDiffusionPipeline.from_ckpt(
    os.path.join(project_path, "models" , "ghostmix_v12.safetensors"),
    torch_dtype=torch.float16,
    # https://github.com/huggingface/diffusers/issues/3212
    # clip_skip
    num_hidden_layers=11,
    scheduler_type="dpm")

embeddings = pipe.load_textual_inversion([os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")])
pipe.to("cuda")
pipe.unet.enable_xformers_memory_efficient_attention()

# pipe.load_lora_weights(os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"))

prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair"
negative_prompt = "EasyNegative"

image = pipe(
    prompt=prompt,
    num_inference_steps=30,
    guidance_scale=7,
    negative_prompt=negative_prompt,
    negative_prompt_embeds=embeddings,
    height=512,
    width=512,
).images[0]
outdir = os.path.join(project_path, "outputs")
image.save(os.path.join(outdir, "test.png"))

