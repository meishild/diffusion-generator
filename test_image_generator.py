import sys, os 
import random

project_path = os.path.abspath(".")
os.environ["TRANSFORMERS_CACHE"] = ".cache/huggingface/transformers"

from datetime import datetime

class TrainProject:
   def __init__(self, train_name, train_repeat) -> None:
      self.base_model = os.path.join(project_path, "models" , "NAI-full.ckpt")
      self.train_name = train_name
      self.train_repeat = train_repeat
      self.work_path = os.path.abspath(".")
      self.project_path = None
      self.train_img_path = None
      self.tagger_img_path = None
      self.sources_img_path = None
      self.checkpoints_path = None
      self.log_path = None
      self.train_config_path = None
   
   def init_project(self, dt=None):
      global project_path
      pn_path = os.path.join(project_path, "train-projects" , self.train_name)
      if not os.path.exists(pn_path):
         os.makedirs(pn_path)
      if dt is None:
         dt = datetime.now().strftime('%Y%m%d%H%M%S')
      pt_path = os.path.join(pn_path, "train-%s" % dt)
      if not os.path.exists(pt_path):
         os.makedirs(pt_path)

      self.project_path = pt_path
      self.sources_img_path = os.path.join(self.work_path, "resources" , "source-images")
      self.checkpoints_path = os.path.join(self.project_path, "checkpoints")
      self.log_path = os.path.join(self.project_path, "logs")
      self.train_config_path = os.path.join(project_path, "resources", "train_config.json")

   def test_cmd_checkpoints(self):
      """
      通过sd_script命令进行生成，由于通过命令，每一次生成都需要重新加载模型
      https://note.com/kohya_ss/n/n2693183a798e

      single LoRA:
      python gen_img_diffusers.py --ckpt name_to_ckpt.safetensors --n_iter 1 --scale 8 --steps 40 \
         --outdir txt2img/samples --xformers --W 512 --H 512 --fp16 --sampler k_euler_a \
         --network_module networks.lora --network_weights lora1.safetensors --network_mul 1.0 \
         --max_embeddings_multiples 3 --clip_skip 2 --batch_size 1 --images_per_prompt 1 \
         --prompt "beautiful scene --n negative prompt"

      two LoRAs:
      python gen_img_diffusers.py --ckpt name_to_ckpt.safetensors --n_iter 1 --scale 8 --steps 40 \
         --outdir txt2img/samples --xformers --W 512 --H 512 --fp16 --sampler k_euler_a \
         --network_module networks.lora networks.lora --network_weights lora1.safetensors lora2.safetensors --network_mul 1.0 0.8 \
         --max_embeddings_multiples 3 --clip_skip 2 --batch_size 1 --images_per_prompt 1 \
         --prompt "beautiful scene --n negative prompt"
      """

      """
      https://github.com/kohya-ss/sd-webui-additional-networks/issues/148

      python .\points\sd_scripts\gen_img_diffusers.py --prompt "1girl" --outdir E:/Develop/sd-train-links/outputs ^
         --H 512 --W 512 --steps 30 --sampler k_dpm_2 --ckpt E:/Develop/sd-train-links/models/NAI-full.ckpt ^
         --network_module networks.lora --network_weights E:/Develop/sd-train-links/outputs/dribbble-design-000001.safetensors ^
         --network_mul 1.0 --seed -1 --clip_skip 2 --fp16 --xformers
      """
      seed = random.randint(0, 0x7FFFFFFF)
      gen_dict = {
         "ckpt" : self.base_model,
         "sampler" : "dpmsolver++",
         "max_embeddings_multiples" : 3,
         "clip_skip": 2,
         "seed" : seed,
         "fp16": True,
         "xformers": True,
      }

      check_points = [name for name in os.listdir(self.checkpoints_path)if name.endswith('.safetensors')]

      for check_point in check_points:                                         
         args_list = []
         for key, value in gen_dict.items():
            if not value:
               continue
            if isinstance(value, bool):
               args_list.append(f"--{key}")
            else:
               args_list.append(f"--{key}={value}")
         args_list.append(f"--network_module=networks.lora")
         args_list.append(f"--network_weights=%s" % os.path.join(self.checkpoints_path, check_point))
         args_list.append(f"--network_mul=0.6")
         args_list.append(f"--sequential_file_name")
         args_list.append(f"--prompt=masterpiece, best quality, 1girl" + " --n " + "bad quality, worst quality, bad anatomy, bad hands")
         # name_space = image_generator.setup_parser().parse_args(args_list)
         # image_generator.main(name_space)
   
   def test_cmd_checkpoints_once(self):
      args = []
      args.append('--prompt "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair --n EasyNegative"')
      args.append("--H 512")
      args.append("--W 512")
      args.append("--steps 150")
      args.append("--fp16")
      args.append("--ckpt %s" % os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
      args.append("--outdir %s" % os.path.join(self.project_path, "images"))
      args.append("--sampler dpmsolver++")
      args.append("--clip_skip 2")
      args.append("--scale 8")
      args.append("--seed 280681258")
      args.append("--xformers")
      args.append("--network_module networks.lora")
      args.append("--network_weights %s" % os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"))
      args.append("--network_mul 0.8")
      args.append("--max_embeddings_multiples 3")
      args.append("--textual_inversion_embeddings %s" % os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors"))
      python = r"python310\python.exe"
      cmd = f"{python} points/sd_scripts/gen_img_diffusers.py %s" % " ".join(args)
      print(cmd)

   def test_checkpoints_once(self):
      from gen_img import GenImages, Txt2ImgParams, NetWorkData

      txt2img= GenImages()
      txt2img.set_dtype("fp16")
      txt2img.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
      txt2img.outdir = os.path.join(self.project_path, "images")
      
      txt2img.xformers = True
      txt2img.max_embeddings_multiples = 3
      txt2img.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]

      prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair"
      negative_prompt = "EasyNegative"
      txt2img.create_pipline()
      # ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a,

      # txt2img.load_vae(os.path.join(project_path, "models", "vae", "animevae.pt"))
      params = Txt2ImgParams(
         sampler="dpmsolver++",
         prompt=prompt,
         negative_prompt=negative_prompt,
         steps=150,
         width=512,
         height=512,
         scale=7,
         seed=280681258,
         clip_skip=2,
         batch_size=1
      )
      
      network = NetWorkData(
            network_module="networks.lora",
            network_weight=os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"), 
            network_mul=0.8,
      )
      network.network_merge = True

      params.networks=[network]
      txt2img.txt2img(params)

   def test_checkpoints_n(self):
      from gen_img import GenImages, Txt2ImgParams, NetWorkData

      txt2img= GenImages()
      txt2img.set_dtype("fp16")
      txt2img.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
      txt2img.outdir = os.path.join(self.project_path, "images")

      txt2img.xformers = True
      txt2img.max_embeddings_multiples = 1
      txt2img.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]

      prompt = "1 girl, cute, solo, beautiful detailed sky, city ,detailed cafe, night, sitting, dating, (smile:1.1),(closed mouth) medium breasts,beautiful detailed eyes,(collared shirt:1.1),pleated skirt,(long hair:1.2),floating hair"
      negative_prompt = "EasyNegative"
      txt2img.create_pipline()

      self.checkpoints_path = "E:\\ai-stable-diffsuion\\LoRA\\lora-train\\outputs\\ghiblistyle\\\dylora"
      check_points = [name for name in os.listdir(self.checkpoints_path)if name.endswith('.safetensors')]

      seed = random.randint(0, 0x7FFFFFFF)
      for check_point in check_points:
         params = Txt2ImgParams(
            sampler="dpmsolver++",
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=50,
            width=512,
            height=512,
            scale=7,
            seed=seed,
            clip_skip=2,
         )
         network = NetWorkData(
            network_module="networks.lora",
            network_weight=os.path.join(self.checkpoints_path, check_point), 
            network_mul=0.6
         )
         network2 = NetWorkData(
            network_module="networks.lora",
            network_weight=os.path.join(project_path, "models", "lora", "JiaranDianaLoraASOUL_v20SingleCostume.safetensors"), 
            network_mul=0.8
         )
         # network.network_merge=True
         # network2.network_merge=True
         params.networks=[network, network2]
         txt2img.txt2img(params)


if __name__ == '__main__':
   from gen_img import GenImages, Txt2ImgParams, NetWorkData

   txt2img = GenImages()
   txt2img.set_dtype("fp16")
   txt2img.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
   txt2img.outdir = os.path.join(project_path, "images")
   
   txt2img.xformers = True
   txt2img.max_embeddings_multiples = 3
   txt2img.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]
   prompt = "1girl, navel, long hair, gloves, breasts, solo, crop top, suspenders, tank top, midriff, black hair, elbow gloves, looking at viewer, earrings, smile, skirt, outdoors, white tank top, fingerless gloves, jewelry, large breasts, black skirt, black gloves, suspender skirt, elbow pads, bare shoulders, taut shirt, taut clothes, parted lips, sky, day, belt, shirt, sports bra, dangle earrings, stomach, swept bangs, low-tied long hair, cleavage, lips"
   negative_prompt = "EasyNegative"
   txt2img.create_pipline()

   # ddim,pndm,lms,euler,euler_a,heun,dpm_2,dpm_2_a,dpmsolver,dpmsolver++,dpmsingle,k_lms,k_euler,k_euler_a,k_dpm_2,k_dpm_2_a,
   # txt2img.load_vae(os.path.join(project_path, "models", "vae", "animevae.pt"))
   params = Txt2ImgParams(
      sampler="dpmsolver++",
      prompt=prompt,
      negative_prompt=negative_prompt,
      steps=50,
      width=512,
      height=512,
      scale=7,
      seed=585790273,
      clip_skip=2,
      batch_size=1
   )
   # seed = random.randint(0, 0x7FFFFFFF)

   lora_path = "E:\\ai-stable-diffsuion\\Cybertronfurnace\\train\\model"
   network = NetWorkData(
         network_module="networks.lora",
         network_weight=os.path.join(lora_path, "gufeng_20230618220730-000019.safetensors"), 
         network_mul=0.8,
   )
   network.network_merge = True
   params.networks = [network]
   txt2img.txt2img(params)