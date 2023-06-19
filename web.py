import gradio as gr
import os
import logging
import random
from gen_img import GenImages, Txt2ImgParams, NetWorkData
import openai
import time

project_path = os.path.abspath(".")
LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

openai.api_key = os.environ["openai_api_key"]
openai.api_base = os.environ["openai_api_base"]

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# Txt2Img WebUI
"""

init_message = f"""可以输入prompt生成图片"""


default_theme_args = dict(
    font=["Source Sans Pro", 'ui-sans-serif', 'system-ui', 'sans-serif'],
    font_mono=['IBM Plex Mono', 'ui-monospace', 'Consolas', 'monospace'],
)

txt2img = GenImages()


def change_mode(mode, history):
    if mode == "txt2img":
        history.append([None, "切换为图像生成模式。"])
        return gr.update(visible=True), gr.update(visible=False), history
    elif mode == "test":
        history.append([None, "切换为测试模式，整体运行时间比较长。"])
        return gr.update(visible=False), gr.update(visible=True), history


def init_model():
    try:
        txt2img.set_dtype("fp16")
        txt2img.set_ckpt(os.path.join(project_path, "models" , "ghostmix_v12.safetensors"))
        txt2img.outdir = os.path.join(project_path, "outputs")
        
        txt2img.xformers = True
        txt2img.max_embeddings_multiples = 3
        txt2img.textual_inversion_embeddings = [os.path.join(project_path, "models", "embeddings", "EasyNegative.safetensors")]

        txt2img.create_pipline()

        reply = """模型已成功加载，可以开始生成"""
        logger.info(reply)
        return reply
    except Exception as e:
        logger.error(e)
        reply = """模型未成功加载，需要重新配置"""
        if str(e) == "Unknown platform: darwin":
            logger.info("该报错可能因为您使用的是 macOS 操作系统，需先下载模型至本地后执行 Web UI，具体方法请参考项目 README 中本地部署方法及常见问题")
        else:
            logger.info(reply)
        return reply

def gen_prompt(query):
    template = f"""StableDiffusion是一款利用深度学习的文生图模型，支持通过使用提示词来产生新的图像，描述要包含或省略的元素。 我在这里引入StableDiffusion算法中的Prompt概念，又被称为提示符。 下面的prompt是用来指导AI绘画模型创作图像的。它们包含了图像的各种细节，如人物的外观、背景、颜色和光线效果，以及图像的主题和风格。这些prompt的格式经常包含括号内的加权数字，用于指定某些细节的重要性或强调。例如，"(masterpiece:1.5)"表示作品质量是非常重要的，多个括号也有类似作用。此外，如果使用中括号，如"(blue hair:white hair:0.3)"，这代表将蓝发和白发加以融合，蓝发占比为0.3。 
以下是用prompt帮助AI模型生成图像的例子：
masterpiece,(bestquality),highlydetailed,ultra-detailed, cold,solo,(1girl),detailedeyes,shinegoldeneyes) (longliverhair) expressionless,(long sleeves,puffy sleeves),(white wings),shinehalo,(heavymetal :1.2),(metaljewelry),cross-lacedfootwear,(chain),(Whitedoves : 1.2)
仿照例子，并不局限于我给你的单词，给出一套详细描述“{query}”的prompt，注意：masterpiece,(bestquality),highlydetailed,ultra-detailed,1 girl,必须放在前面，prompt不能超过80个。直接开始给出prompt不需要用自然语言描述。
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content": template}])
    prompt = response['choices'][0]['message']['content']
    return prompt

def gen_image(query, history, mode, txt2img_type, batch_size=1):
    prompt = query
    if txt2img_type == "chatgpt":
        prompt = gen_prompt(query)
    
    if mode == "txt2img":
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
            batch_size=batch_size
        )
        images_path = txt2img.txt2img(params)
        
        image_md = f"{prompt}" + "\n".join(["![{}](/file=%s)" % path for path in images_path])

        history.append([query, image_md])
        return history, query
    
    elif mode == "img2img":
        history.append([query, "暂时不支持图生图"])
        return history, query 

def get_torch():
    import torch
    try:
        ver = torch.__long_version__
    except Exception:
        ver = torch.__version__
    return ver

def get_gpu():
    import torch

    device = None
    cdua = None
    cudnn = None
    if not torch.cuda.is_available():
        try:
            device = f'{torch.xpu.get_device_name(torch.xpu.current_device())} ({str(torch.xpu.device_count())})'
        except Exception:
            pass
    else:
        try:
            if torch.version.cuda:
                device = f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())}) ({torch.cuda.get_arch_list()[-1]})'
                cuda = torch.version.cuda
                cudnn = torch.backends.cudnn.version()
            elif torch.version.hip:
                device = f'{torch.cuda.get_device_name(torch.cuda.current_device())} ({str(torch.cuda.device_count())})'
            else:
                device = 'unknown'
        except Exception:
            pass
    
    return device, cuda, cudnn

def get_gpu_mem_info(gpu_id=0):
    """
    根据显卡 id 获取显存使用信息, 单位 MB
    :param gpu_id: 显卡 ID
    :return: total 所有的显存，used 当前使用的显存, free 可使用的显存
    """
    import pynvml
    pynvml.nvmlInit()
    if gpu_id < 0 or gpu_id >= pynvml.nvmlDeviceGetCount():
        print(r'gpu_id {} 对应的显卡不存在!'.format(gpu_id))
        return 0, 0, 0

    handler = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handler)
    total = round(meminfo.total / 1024 / 1024, 2)
    used = round(meminfo.used / 1024 / 1024, 2)
    free = round(meminfo.free / 1024 / 1024, 2)
    return total, used, free

def gen_images_test(history, size = 10):
    import xformers
    import platform
    import transformers

    prompt = "1 girl, cute, solo"
    params = Txt2ImgParams(
            sampler="euler",
            prompt=prompt,
            negative_prompt="EasyNegative",
            steps=10,
            width=512,
            height=512,
            scale=7,
            seed=random.randint(0, 0x7FFFFFFF),
            clip_skip=2,
            batch_size=1
        )
    txt2img.txt2img(params)

    start = time.time()
    params.steps = 150
    used_gpu_mem = 0

    for _ in range(0, size, 1):
        txt2img.txt2img(params)
        total, used, free = get_gpu_mem_info()
        used_gpu_mem += used

    used_ts = time.time() - start
    avg_speed = "%.2f" % float(params.steps * size / used_ts)
    device, cuda, cudnn = get_gpu()
    tmp = f"""环境信息：
arch: {platform.machine()}
cpu: {platform.processor()}
system: {platform.system()}
python: {platform.python_version()}
torch: {get_torch()}
gpu: {device}
cdua: {cuda}
cudnn: {cudnn}
xformers: {xformers.__version__}
transformers: {transformers.__version__}
---
测试数据：
sampler: {params.sampler}
prompt: {params.prompt}
negative_prompt: {params.negative_prompt}
steps: {params.steps}
image_size: {params.width}x{params.height}
test_size: {size}
---
测试结果：
all_steps: {params.steps * size}
all_time: {used_ts:.2f} s
avg_speed: {avg_speed} it/s
avg_gpu_mem: {used_gpu_mem / size} MB
"""

    history.append([None, tmp])
    return history 

# 初始化消息
model_status = init_model()

with gr.Blocks(css=block_css, theme=gr.themes.Default(**default_theme_args)) as demo:
    gr.Markdown(webui_title)
    
    with gr.Tab("对话生成"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message], [None, model_status]],
                                     elem_id="chat-box",
                                     show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,
                                   placeholder="请输入promot内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["txt2img", "img2img", "test"],
                                label="请选择生成模式",
                                value="txt2img",)
                
                gimg_mode = gr.Accordion("选择输入模式")
                test_mode = gr.Accordion("测试模式", visible=False)
                mode.change(fn=change_mode, inputs=[mode, chatbot], outputs=[gimg_mode, test_mode, chatbot])

                with gimg_mode:
                    txt2img_type = gr.Radio(["prompt", "chatgpt"],
                                label="输出内容模式",
                                value="prompt",)
                    batch_size = gr.Radio([1, 2, 4],
                                label="每批生成数量",
                                value=1,)
                with test_mode:
                    test_type = gr.Radio(["生成速度"],
                                label="测试模式",
                                value="生成速度")
                    test_button = gr.Button("启动测试")
                    
                    test_button.click(fn=gen_images_test,
                                     inputs=[chatbot],
                                     outputs=[chatbot])

            query.submit(gen_image,[query, chatbot, mode, txt2img_type, batch_size], [chatbot, query])
(demo
 .queue(concurrency_count=3)
 .launch(server_name='0.0.0.0',
         server_port=8080,
         show_api=False,
         share=False,
         inbrowser=False))
