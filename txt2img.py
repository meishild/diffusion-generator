import os 

project_path = os.path.abspath(".")

if __name__ == '__main__':
   from gen_img import GenImages, Txt2ImgParams, NetWorkData

   txt2img = GenImages()
   txt2img.set_dtype("fp16")
   # txt2img.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
   txt2img.set_ckpt("E:\\ai-stable-diffsuion\\SDXL0.9\\13G version\\sd_xl_base_0.9.safetensors")
   txt2img.outdir = os.path.join(project_path, "outputs")
   
   txt2img.xformers = True
   txt2img.max_embeddings_multiples = 3
   txt2img.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]

   txt2img.create_pipline()

   # ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a,
   txt2img.load_vae(os.path.join(project_path, "models", "vae", "animevae.pt"))
   
   # network = NetWorkData(
   #       network_module="networks.lora",
   #       network_weight=os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"), 
   #       network_mul=0.8,
   # )
   # network.network_merge = True

   prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, medium breasts,beautiful detailed eyes"
   negative_prompt = "EasyNegative"
   params = Txt2ImgParams(
      sampler="euler",
      prompt=prompt,
      negative_prompt=negative_prompt,
      steps=30,
      width=768,
      height=768,
      scale=7,
      seed=585790273,
      clip_skip=2,
      batch_size=1
   )
   # seed = random.randint(0, 0x7FFFFFFF)

   txt2img.txt2img(params)