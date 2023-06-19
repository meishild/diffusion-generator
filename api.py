import os
from sanic import Sanic, response
from sanic.response import text
from gen_img import GenImages, Txt2ImgParams, NetWorkData

app = Sanic("Txt2Img")
project_path = os.path.abspath(".")

@app.before_server_start
async def setup(app, loop):
    gen_images = GenImages()
    gen_images.set_dtype("fp16")
    gen_images.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
    gen_images.outdir = os.path.join(project_path, "outputs")
    
    gen_images.xformers = True
    gen_images.max_embeddings_multiples = 3
    gen_images.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]
 
    gen_images.create_pipline()
 
    # ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a,
    # txt2img.load_vae(os.path.join(project_path, "models", "vae", "animevae.pt"))
    app.ctx.gen_images=gen_images

@app.get("/demo")
async def demo(request):
    network = NetWorkData(
          network_module="networks.lora",
          network_weight=os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"), 
          network_mul=0.8,
    )
    network.network_merge = True

    prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair"
    negative_prompt = "EasyNegative"
    params = Txt2ImgParams(
        sampler="dpmsolver++",
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=30,
        width=512,
        height=512,
        scale=7,
        seed=585790273,
        clip_skip=2,
        batch_size=1
    )
    # seed = random.randint(0, 0x7FFFFFFF)
    gen_images = request.app.ctx.gen_images
    images_path = gen_images.txt2img(params)
    return await response.file(images_path[0])

@app.post("/txt2img")
async def txt2img(request):
    import random
    
    json_data = request.json
    prompt = json_data["prompt"]

    # prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair"
    negative_prompt = "EasyNegative"
    params = Txt2ImgParams(
        sampler="dpmsolver++",
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=30,
        width=512,
        height=512,
        scale=7,
        seed=random.randint(0, 0x7FFFFFFF),
        clip_skip=2,
        batch_size=1
    )
    gen_images = request.app.ctx.gen_images
    images_path = gen_images.txt2img(params)
    return await response.file(images_path[0])

if __name__ == '__main__':
    app.run(port=5001, host='0.0.0.0',debug=False)
 