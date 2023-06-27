# Diffusion Generator

## diffuser实现SD需要支持的功能

### 基础能力

* 加载模型：至少支持SD 1.5基础模型以及相关的其他预训练模型并且支持基于safetensors的封装。

* prompt：支持至少70token的prompt，更高长度的promt可以后续支撑。

* negative_prompt: 支持返乡prompt，并且基于这个可以支持negative_prompt的embeddings。

* scale：图像发散程度参数。

* 图像参数：宽度、高度。

* skip：跳过层数，或者支持隐藏层。

* seed：至少可以定义引导seed。

* vae：可以load其他vae替换safetensors的vae。

* xformers\sdp：至少支持一种，xformers兼容性更强，sdp在tourch2.0后默认支持。

* 性能速度：生成速度需要和直接使用stable diffusion webui 要相同。

### lora能力

* 可以加载lora。