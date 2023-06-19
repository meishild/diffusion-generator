这是一个基于Diffusers的推理（图像生成）脚本，支持SD 1.x和2.x模型，以及在此存储库中学习的LoRA、ControlNet（仅在v1.0中进行了测试）等。通过命令行使用。

# 概要

* 该推理（图像生成）脚本基于Diffusers（v0.10.2）。
* 支持SD 1.x和2.x（base/v-parameterization）模型。
* 支持txt2img、img2img、inpainting等应用场景。
* 支持交互模式、从文件中加载提示、连续生成等。
* 可以指定每行提示生成的图片数量。
* 可以指定总重复次数。
* 支持`fp16`和`bf16`。
* 支持xformers，可以进行快速生成。
   * 使用xformers进行省内存生成，但与Automatic 1111的Web UI相比，优化程度不够，因此在生成512*512大小的图片中使用约6GB VRAM。
* 提示扩展到225个令牌。支持负面提示和权重调整。
* 支持Diffusers的多种采样器（比Web UI少）。
* 支持文本编码器clip skip（使用最后n层的输出）。
* 支持单独加载VAE。
* 支持CLIP Guided Stable Diffusion、VGG16 Guided Stable Diffusion、Highres. fix、upscale等。
   * Highres. fix是一种自己实现的方法，没有完全确认Web UI的实现，因此输出结果可能有所不同。
* 支持LoRA。支持应用率指定、同时使用多个LoRA、权重合并。
   * 无法在文本编码器和U-Net中指定不同的适用率。
* 支持Attention Couple。
* 支持ControlNet v1.0。
* 不能在中途更改模型，但可以组合批处理文件来解决。
* 增加了一些我个人需要的功能。

由于在添加功能时并非对所有测试都进行了测试，因此可能会导致早期功能中的问题，某些功能无法正常工作。如果有任何问题，请告诉我。

# 基本用法

## 通过对话模式生成图像

请按照以下方式输入：

```batchfile
python gen_img_diffusers.py --ckpt <模型名称> --outdir <图像输出路径> --xformers --fp16 --interactive
```

使用`--ckpt`选项指定模型（稳定扩散的checkpoint文件或Diffusers模型文件夹），使用`--outdir`选项指定生成图像的输出文件夹。

使用`--xformers`选项指定使用xformers（如果不使用xformers，请忽略此选项）。使用`--fp16`选项进行单精度（fp16）推理。您还可以在RTX 30系GPU上使用`--bf16`选项进行bfloat16推理。

使用`--interactive`选项启用交互模式。

要使用Stable Diffusion 2.0（或其追加训练模型），请添加`--v2`选项。如果要使用v参数化模型（`768-v-ema.ckpt`和其追加训练模型），请再添加`--v_parameterization`选项。

如果指定`--v2`时出错，模型加载时会出现错误。指定`--v_parameterization`时，将显示棕色图像。

当出现`Type prompt:`时，请输入提示。

![image](https://user-images.githubusercontent.com/52813779/235343115-f3b8ac82-456d-4aab-9724-0cc73c4534aa.png)

※ 如果未显示图像并出现错误，则可能已安装不带屏幕显示功能的OpenCV。请安装通常的OpenCV `pip install opencv-python`，或使用`--no_preview`选项停止显示图像。

选择图像窗口，然后按任意键关闭窗口，就可以输入下一个提示了。按下Ctrl+Z和Enter，以关闭脚本。


## 单一提示批量生成图像

输入如下（实际上在一行中输入）：

```batchfile
python gen_img_diffusers.py --ckpt <模型名称> --outdir <图像输出目录> 
    --xformers --fp16 --images_per_prompt <生成数量> --prompt "<提示>"
```

使用`--images_per_prompt`选项指定每个提示生成的图片数量。使用`--prompt`选项指定提示。如果包含空格，请使用双引号括起来。

您可以使用`--batch_size`选项指定批处理大小（请参见后面）。

## 从文件中读取提示并进行批量生成

输入如下：

```batchfile
python gen_img_diffusers.py --ckpt <模型名称> --outdir <图像输出目录> 
    --xformers --fp16 --from_file <提示文件名称>
```

使用`--from_file`选项指定包含提示的文件。每行一个提示。您可以使用`--images_per_prompt`选项指定每行生成的图像数量。


## 使用负面提示和加权

在提示选项（在提示中指定为 `--x`，稍后会详细介绍）中写入 `--n`，则以下内容将成为负面提示。

此外，您可以使用与AUTOMATIC1111的Web UI相同的`（）`、`[]`、`（xxx:1.3）`等符号进行加权（实现是从Diffusers的[Long Prompt Weighting Stable Diffusion](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#long-prompt-weighting-stable-diffusion)中复制的）。

您可以在命令行上指定提示，也可以从文件中读取提示。

![image](https://user-images.githubusercontent.com/52813779/235343128-e79cd768-ec59-46f5-8395-fce9bdc46208.png)


# 主要选项

## 指定模型

- `--ckpt <模型名称>`：指定模型名称。 `--ckpt`选项是必需的。可以指定稳定扩散的checkpoint文件、Diffusers模型文件夹、Hugging Face模型ID。

- `--v2`：在使用Stable Diffusion 2.x系列模型时指定。对于1.x系列，则无需指定。

- `--v_parameterization`：在使用使用v-parameterization的模型时指定（例如，`768-v-ema.ckpt`及其附加学习模型，Waifu Diffusion v1.5等）。

如果指定错误，则在模型加载时会出错。 如果未正确指定，则显示褐色图像。

- `--vae`：指定要使用的VAE。默认情况下，将使用模型内的VAE。

## 图像生成和输出

- `--interactive`：以交互模式运行。输入提示后，将生成图像。

- `--prompt <提示>`：指定提示。如果提示包含空格，请使用双引号括起来。

- `--from_file <提示文件名>`：指定包含提示的文件。请按1行1个提示进行编写。可以使用提示选项（如下所述）指定图像大小和guidance scale。

- `--W <图像宽度>`：指定图像宽度。默认为`512`。

- `--H <图像高度>`：指定图像高度。默认为`512`。

- `--steps <采样步数>`：指定采样步数。默认为`50`。

- `--scale <guidance scale>`：指定uncoditioning的guidance scale。默认为`7.5`。

- `--sampler <采样器名称>`：指定采样器。默认为`ddim`。可以使用Diffusers提供的ddim、pndm、dpmsolver、dpmsolver+++、lms、euler、euler_a，也可以使用k_lms、k_euler、k_euler_a来指定。

- `--outdir <图像输出文件夹>`：指定图像输出文件夹。

- `--images_per_prompt <生成数量>`：指定每个提示生成的图像数量。默认为`1`。

- `--clip_skip <跳过层数>`：指定要使用CLIP的第几层开始。如果省略，默认使用最后一层。

- `--max_embeddings_multiples <倍数>`：指定要将CLIP的输入输出长度乘以多少倍。默认为75。例如，指定为3，输入输出长度将变为225。

- `--negative_scale`：分别指定每个uncoditioning的guidance scale。这是参考[gcem156的文章](https://note.com/gcem156/n/ne9a53e4a6f43)实现的。

## 调整内存使用量和生成速度：

- `--batch_size <批量大小>`：指定批量大小。默认为`1`。较大的批量大小会消耗更多内存，但可以更快地生成图像。

- `--vae_batch_size <VAE批量大小>`：指定VAE批量大小。默认与批量大小相同。由于VAE消耗更多内存，在去噪之后（当步骤达到100％时）可能会出现内存短缺。在这种情况下，应该减小VAE批量大小。

- `--xformers`：指定使用xformers。

- `--fp16`：使用单精度（fp16）进行推断。如果不同时指定`fp16`和`bf16`选项，则使用单精度（fp32）进行推断。

- `--bf16`：使用bf16（bfloat16）进行推断。仅适用于RTX 30系列GPU。在RTX 30系列以外的GPU上使用`--bf16`选项将会报错。与`fp16`相比，`bf16`的推理结果可能更不容易出现NaN（黑色图像）的情况。

## 使用附加网络（如LoRA）：

- `--network_module`：指定要使用的附加网络。例如，对于LoRA，请指定`--network_module networks.lora`。如果要使用多个LoRA，请指定`--network_module networks.lora networks.lora networks.lora`。

- `--network_weights`：指定要使用的附加网络的权重文件。例如，指定`--network_weights model.safetensors`。如果要使用多个LoRA，请指定`--network_weights model1.safetensors model2.safetensors model3.safetensors`。请确保指定的文件数与`--network_module`中指定的数量相同。

- `--network_mul`：指定将要使用的附加网络的权重应增加多少倍。默认值为`1`。例如，指定`--network_mul 0.8`。如果要使用多个LoRA，请指定`--network_mul 0.4 0.5 0.7`。请确保指定的参数数与`--network_module`中指定的数量相同。

- `--network_merge`：预先使用`--network_mul`指定的权重合并要使用的附加网络。这与`--network_pre_calc`不能同时使用。这将禁用选项`--am`和Regional LoRA，但生成速度将提高到LoRA未使用时相同的水平。

- `--network_pre_calc`：预先为每次生成计算要使用的附加网络的权重。`--am`选项可用于提示。这将提高生成速度，但需要在生成之前计算权重，这需要一些时间，并且会稍微增加内存使用量。这在使用Regional LoRA时无效。

# 主要选项的指定示例

以下是在相同提示符下使用批量大小4生成64张图片的示例。

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 64 
    --prompt "beautiful flowers --n monochrome"
```

以下是在文件中指定的提示符，以批量大小4一次生成10张图像的示例。

```batchfile
python gen_img_diffusers.py --ckpt model.ckpt --outdir outputs 
    --xformers --fp16 --W 512 --H 704 --scale 12.5 --sampler k_euler_a 
    --steps 32 --batch_size 4 --images_per_prompt 10 
    --from_file prompts.txt
```

以下是Textual Inversion和LoRA使用的示例：

```batchfile
python gen_img_diffusers.py --ckpt model.safetensors 
    --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --fp16 --sampler k_euler_a 
    --textual_inversion_embeddings goodembed.safetensors negprompt.pt 
    --network_module networks.lora networks.lora 
    --network_weights model1.safetensors model2.safetensors 
    --network_mul 0.4 0.8 
    --clip_skip 2 --max_embeddings_multiples 1 
    --batch_size 8 --images_per_prompt 1 --interactive
```


# Prompt 选项

在命令提示符中，可以使用诸如 `--n` 的“双短横线+ n 个字母”的选项来指定各种选项。无论是从交互模式、命令行还是文件指定提示，都有效。

请在" --n "选项的前后留出空格。

- `--n`：指定负面提示。

- `--w`：指定图像宽度。覆盖命令行中的指定。

- `--h`：指定图像高度。覆盖命令行中的指定。

- `--s`：指定步骤数。覆盖命令行中的指定。

- `--d`：指定这个图像的随机种子。如果您使用 "--images_per_prompt"，请使用逗号分隔指定多个种子，例如 "--d 1,2,3,4"。
    ※由于各种原因，即使使用相同的随机种子，生成的图像也可能与 Web UI 中的图像不同。

- `--l`：指定 guidance scale。覆盖命令行中的指定。

- `--t`：指定 img2img（稍后指定）的强度。覆盖命令行中的指定。

- `--nl`：指定负面提示的 guidance scale（稍后指定）。覆盖命令行中的指定。

- `--am`：指定附加网络的权重。覆盖命令行中的指定。如果使用多个附加网络，请使用逗号分隔它们，例如 `--am 0.8,0.5,0.3` 。

Note: 如果指定了这些选项，则批处理可能会在小于批处理大小的情况下执行（这是因为这些值不同，因此无法进行批量生成）。 （不用太在意，但是如果从文件中加载提示并进行生成，则将这些值放在相同的提示中将更有效率。）

例：
```
(masterpiece, best quality), 1girl, in shirt and plated skirt, standing at street under cherry blossoms, upper body, [from below], kind smile, looking at another, [goodembed] --n realistic, real life, (negprompt), (lowres:1.1), (worst quality:1.2), (low quality:1.1), bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, normal quality, jpeg artifacts, signature, watermark, username, blurry --w 960 --h 640 --s 28 --d 1
```

![image](https://user-images.githubusercontent.com/52813779/235343446-25654172-fff4-4aaf-977a-20d262b51676.png)

# img2img

## 选项

- `--image_path`：指定用于 img2img 的图像。指定方式为 `--image_path template.png`。如果指定文件夹，则按顺序使用该文件夹中的图像。

- `--strength`：指定 img2img 的强度。指定方式为 `--strength 0.8`。默认为 `0.8`。

- `--sequential_file_name`：指定文件名是否按连续编号。如果指定，则生成的文件名将从`im_000001.png`开始按连续编号。

- `--use_original_file_name`：指定后，生成的文件名将与原始文件名相同。

## 命令行示例

```batchfile
python gen_img_diffusers.py --ckpt trinart_characters_it4_v1_vae_merged.ckpt 
    --outdir outputs --xformers --fp16 --scale 12.5 --sampler k_euler --steps 32 
    --image_path template.png --strength 0.8 
    --prompt "1girl, cowboy shot, brown hair, pony tail, brown eyes, 
          sailor school uniform, outdoors 
          --n lowres, bad anatomy, bad hands, error, missing fingers, cropped, 
          worst quality, low quality, normal quality, jpeg artifacts, (blurry), 
          hair ornament, glasses" 
    --batch_size 8 --images_per_prompt 32
```

指定`--image_path`选项文件夹时，它会顺序读取该文件夹中的图像。生成的图像数量不是图像数量，而是提示数量，请使用`--images_per_prompt`选项来指定图像数量和提示数量。

文件将按文件名排序进行读取。但请注意它按字符串顺序排序（不是`1.jpg→2.jpg→10.jpg`，而是`1.jpg→10.jpg→2.jpg`顺序），因此请进行0填充等处理（例如`01.jpg→02.jpg→10.jpg`）。

## 利用img2img进行upscale

当在img2img时使用命令行选项`--W`和`--H`指定生成图像的大小时，将对原始图像进行调整大小处理，然后再进行img2img处理。

此外，如果img2img的原始图像是通过此脚本生成的图像，则可以省略提示。这样，从元数据中获取提示并将其直接使用。这样，将只执行Highres. fix的第二阶段操作。

## img2img时的inpainting

可以指定图像和蒙板图像进行inpainting（它不支持inpainting模型，只是针对蒙板区域进行img2img）。

选项如下：

- `--mask_image`：指定蒙板图像。如果像`--img_path`一样指定文件夹，则将顺序使用该文件夹中的图像。

蒙板图像是灰度图像，如果对白色进行了inpainting，则对其进行处理。推荐将边界进行渐变处理，这样感觉会更加平滑。

![image](https://user-images.githubusercontent.com/52813779/235343795-9eaa6d98-02ff-4f32-b089-80d1fc482453.png)

# 其他功能

## Textual Inversion

可以使用`--textual_inversion_embeddings`选项指定要使用的embeddings（可以指定多个）。通过在提示中使用不带扩展名的文件名，可以使用该embeddings（与Web UI的使用方式相同）。您还可以在Negative prompt内使用它们。

可以使用在此存储库中训练的Textual Inversion模型以及在Web UI中训练的Textual Inversion模型（不支持图像嵌入）作为模型。

## Extended Textual Inversion

请使用`--XTI_embeddings`选项代替`--textual_inversion_embeddings`选项。使用方式与`--textual_inversion_embeddings`相同。

## Highres. fix

这是AUTOMATIC1111的Web UI中的一种类似功能（由于我们自己的实现可能有所不同）。它首先生成一个小的图像，然后使用它执行img2img，同时防止整个图像出现问题，同时生成高分辨率的图像。

第二阶段的步数是从`--steps`和`--strength`选项（`steps * strength`）中计算出来的。

它无法与img2img一起使用。

以下选项可用：

- `--highres_fix_scale`：启用Highres. fix并在乘数中指定生成第1阶段图像的大小。例如，如果您首先生成512x512的图像，然后将其输出到1024x1024，则应指定`--highres_fix_scale 0.5`。请注意，这是Web UI规范的倒数。

- `--highres_fix_steps`：指定第1阶段图像中的步数。默认值为`28`。

- `--highres_fix_save_1st`：指定是否保存第1阶段图像。

- `--highres_fix_latents_upscaling`：如果指定了，则在生成第2阶段图像时以潜在为基础将第1阶段图像upscale（仅支持双线性）。否则，图像将使用LANCZOS4进行upscale。

- `--highres_fix_upscaler`：对第2阶段使用任何放大器。目前，仅支持 `--highres_fix_upscaler tools.latent_upscaler`。

- `--highres_fix_upscaler_args`：指定传递给`--highres_fix_upscaler`指定的upscale器的参数。
对于`tools.latent_upscaler`情况，请使用例如 `--highres_fix_upscaler_args "weights=D:\Work\SD\Models\others\etc\upscaler-v1-e100-220.safetensors"` 来指定重量文件。

以下是一个示例命令行。

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 1024 --H 1024 --batch_size 1 --outdir ../txt2img 
    --steps 48 --sampler ddim --fp16 
    --xformers 
    --images_per_prompt 1  --interactive 
    --highres_fix_scale 0.5 --highres_fix_steps 28 --strength 0.5
```

## ControlNet

目前仅验证ControlNet 1.0的功能。预处理仅支持Canny算法。

以下是可用选项：

- `--control_net_models`：指定ControlNet模型文件。您可以指定多个模型，系统将在每个步骤中切换它们（这与Web UI的ControlNet扩展实现不同）。支持diff和普通模型。

- `--guide_image_path`：指定用于ControlNet的引导图像。与 `--img_path` 相同，您可以指定文件夹并按顺序使用其中的图像。对于不是Canny算法的其他模型，请先执行预处理。

- `--control_net_preps`：指定ControlNet的预处理方式。您可以像 `--control_net_models` 一样指定多个预处理方式。目前仅支持canny。如果在目标模型中未使用预处理，则指定 `none`。

 对于Canny，您可以像 `--control_net_preps canny_63_191` 这样使用下划线分隔1和2的阈值。

- `--control_net_weights`：指定应用ControlNet时的权重（使用 `1.0` 是默认值，`0.5` 表示在应用时只有一半的影响力）。您可以像 `--control_net_models` 一样指定多个权重。

- `--control_net_ratios`：指定应用ControlNet的步骤范围。如果指定为 `0.5`，则仅在步骤数的一半内应用ControlNet。您可以像 `--control_net_models` 一样指定多个比率。

以下是命令行示例。

```batchfile
python gen_img_diffusers.py --ckpt model_ckpt --scale 8 --steps 48 --outdir txt2img --xformers 
    --W 512 --H 768 --bf16 --sampler k_euler_a 
    --control_net_models diff_control_sd15_canny.safetensors --control_net_weights 1.0 
    --guide_image_path guide.png --control_net_ratios 1.0 --interactive
```

Attention Couple + Reginal LoRA 是一个功能强大的模型，可以将提示分成多个部分，并指定将每个提示应用于图像的哪个区域。虽然没有单独的选项，但您可以通过 `mask_path` 和提示来指定。

首先，在提示中使用 `AND` 定义多个部分。可以为前三部分指定区域，而后面的部分将应用于整个图像。负面提示将应用于整个图像。

以下示例使用 `AND` 定义了三个部分：

```
shs 2girls, looking at viewer, smile AND bsb 2girls, looking back AND 2girls --n bad quality, worst quality
```

接下来，您需要准备一个掩模图像。掩模图像应该是一个彩色图像，其中RGB的每个通道对应于提示中使用 `AND` 分割的部分。如果某个通道的值全为0，则该部分将应用于整个图像。

在上面的示例中，R通道对应于`shs 2girls, looking at viewer, smile`，G通道对应于`bsb 2girls, looking back`，B通道对应于`2girls`。如果使用以下掩模图像，则没有指定B通道，因此`2girls`将适用于整个图像。

![image](https://user-images.githubusercontent.com/52813779/235343061-b4dc9392-3dae-4831-8347-1e9ae5054251.png)

您可以通过 `--mask_path` 指定掩模图像。目前只支持一个图像。指定的图像将自动重新调整大小并应用。

您也可以将其与ControlNet结合使用（建议将其与ControlNet结合使用以进行详细的位置指定）。

如果指定LoRA，则 `--network_weights` 中指定的多个LoRA将分别对应于 AND 的每个部分。当前的限制是，LoRA的数量必须等于 AND 部分的数量。

## CLIP Guided Stable Diffusion

这是从 [Diffusers社区示例](https://github.com/huggingface/diffusers/blob/main/examples/community/README.md#clip-guided-stable-diffusion) 复制并修改的自定义流程，除了普通的提示生成之外，还使用更大的CLIP获取提示文本的文本特征，并在生成过程中控制生成的图像接近这些文本特征（这是我的粗略理解）。由于使用了更大的CLIP，VRAM使用量会增加（即使是512*512的图像，VRAM 8GB也可能会有困难），生成时间也会增加。

可用的采样器仅为 DDIM、PNDM 和 LMS。

您可以通过在 `--clip_guidance_scale` 选项中指定一个数值来确定 CLIP 特征对图像的影响程度。在前面的示例中，它设置为 100，因此建议从该值开始进行调整。

默认情况下，将传递给CLIP的前75个令牌（不包括加权特殊字符）。通过提示的 `--c` 选项，您可以指定要传递给CLIP而不是普通提示的文本（例如，CLIP可能无法识别DreamBooth的标识符（identifier）或模型特有单词如“1girl”，因此省略这些单词可能更好）。 

以下是命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt v1-5-pruned-emaonly.ckpt --n_iter 1 
    --scale 2.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img --steps 36  
    --sampler ddim --fp16 --opt_channels_last --xformers --images_per_prompt 1  
    --interactive --clip_guidance_scale 100
```

## CLIP Image Guided Stable Diffusion

这是一个CLIP图像引导的稳定扩散功能，不是文本，而是向CLIP传递另一张图像，以控制生成物接近其特征量。请使用`--clip_image_guidance_scale`选项来指定应用量的数值，使用`--guide_image_path`选项来指定用作引导的图像（文件或文件夹）。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py  --ckpt trinart_characters_it4_v1_vae_merged.ckpt
    --n_iter 1 --scale 7.5 --W 512 --H 512 --batch_size 1 --outdir ../txt2img 
    --steps 80 --sampler ddim --fp16 --opt_channels_last --xformers 
    --images_per_prompt 1  --interactive  --clip_image_guidance_scale 100 
    --guide_image_path YUKA160113420I9A4104_TP_V.jpg
```

### VGG16 Guided Stable Diffusion

这是一个生成图像接近指定图像的功能。除了常规提示生成之外，它还可以获取VGG16特征并控制生成的图像接近指定的引导图像。建议在img2img中使用此功能（通常的生成会使图像模糊）。这是一种借鉴了CLIP Guided Stable Diffusion机制的自定义功能。另外，这个想法来自利用VGG进行风格转换。

可选择的采样器仅限于DDIM、PNM和LMS。

使用`--vgg16_guidance_scale`选项指定应该在多大程度上反映VGG16特征，推荐从100开始尝试并逐步增加或减少。请使用`--guide_image_path`选项来指定用作引导的图像（文件或文件夹）。

如果要批量转换多个图像，并使用原始图像作为引导图像，则可以将`--guide_image_path`和`--image_path`指定为相同的值。

以下是命令行示例：

```batchfile
python gen_img_diffusers.py --ckpt wd-v1-3-full-pruned-half.ckpt 
    --n_iter 1 --scale 5.5 --steps 60 --outdir ../txt2img 
    --xformers --sampler ddim --fp16 --W 512 --H 704 
    --batch_size 1 --images_per_prompt 1 
    --prompt "picturesque, 1girl, solo, anime face, skirt, beautiful face 
        --n lowres, bad anatomy, bad hands, error, missing fingers, 
        cropped, worst quality, low quality, normal quality, 
        jpeg artifacts, blurry, 3d, bad face, monochrome --d 1" 
    --strength 0.8 --image_path ..\src_image
    --vgg16_guidance_scale 100 --guide_image_path ..\src_image 
```

使用`--vgg16_guidance_layer`选项可以指定用于提取VGG16特征的层号（默认为20，即conv4-2的ReLU）。较高层表示风格，较低层表示内容。

![image](https://user-images.githubusercontent.com/52813779/235343813-3c1f0d7a-4fb3-4274-98e4-b92d76b551df.png)

以下是其他的选项：

- `--no_preview`: 在交互模式下不显示预览图像。如果未安装OpenCV，或者希望直接查看输出文件，请使用此选项。

- `--n_iter`: 指定生成的迭代次数。默认值为1。如果从文件中加载提示并希望进行多次生成，则应指定此选项。

- `--tokenizer_cache_dir`: 指定令牌化器的缓存目录。（进行中）

- `--seed`: 指定随机数种子。对于单个生成的图像，这是该图像的种子；对于多个生成的图像，这是生成每个图像所需的随机数种子（使用`--from_file`生成多个图像时，指定`--seed`选项可以确保每个图像都具有相同的种子）。

- `--iter_same_seed`: 如果提示中没有指定随机数种子，则在`--n_iter`的迭代内使用相同的种子。这在使用`--from_file`指定多个提示进行比较时非常有用，可以使种子在几个提示之间统一。

- `--diffusers_xformers`: 使用Diffuser的xformers功能。

- `--opt_channels_last`: 在推断时将张量通道放置在最后。这可能会提高性能。

- `--network_show_meta`: 显示附加网络的元数据。

