# ComfyUI_DiffuEraser
[DiffuEraser](https://github.com/lixiaowen-xw/DiffuEraser) is a diffusion model for video Inpainting, running fully on the ComfyUI-native stack in this plugin.

# Updates
* 改变封装方式，避免diffuser和transformer版本更迭带来的问题/ change the packaging method to avoid problems caused by the version updates of diffuser and transformer.

# 1. Installation
In the `./ComfyUI/custom_nodes` directory:
```
git clone https://github.com/smthemex/ComfyUI_DiffuEraser.git
```
# 2. Requirements  
```
pip install -r requirements.txt
```

# 3. Models
* VAE: [sd-vae-ft-mse](https://huggingface.co/stabilityai/sd-vae-ft-mse/tree/main) → `models/vae/`
* CLIP-l: any standard ComfyUI `clip_l.safetensors` → `models/clip/`
* PCM LoRA: [PCM_Weights/sd15](https://huggingface.co/wangfuyun/PCM_Weights/tree/main/sd15)
  e.g. `pcm_sd15_smallcfg_2step_converted.safetensors` → `models/loras/`
* ProPainter [ProPainter address](https://github.com/sczhou/ProPainter/releases/tag/v0.1.0) # below example
* BrushNet + motion UNet: [lixiaowen/diffuEraser](https://huggingface.co/lixiaowen/diffuEraser/tree/main) → `models/DiffuEraser/`
```
--  ComfyUI/models/vae
    |-- sd-vae-ft-mse.safetensors
--  ComfyUI/models/clip
    |-- clip_l.safetensors
--  ComfyUI/models/loras
    |-- pcm_sd15_smallcfg_2step_converted.safetensors
--  ComfyUI/models/DiffuEraser
     |-- brushnet
        |-- config.json
        |-- diffusion_pytorch_model.safetensors
     |-- unet_main
        |-- config.json
        |-- diffusion_pytorch_model.safetensors
     |-- propainter
        |-- ProPainter.pth
        |-- raft-things.pth
        |-- recurrent_flow_completion.pth
```
* Optional video-to-mask (RMBG / BiRefNet), auto-downloaded or local:
```
-- any/path/briaai/RMBG-2.0        # config.json, model.safetensors, birefnet.py, BiRefNet_config.py
-- any/path/ZhengPeng7/BiRefNet    # + handler.py
```

# 4 Example
![](https://github.com/smthemex/ComfyUI_DiffuEraser/blob/main/example_workflows/example.png)
* use single mask (old version) 
![](https://github.com/smthemex/ComfyUI_DiffuEraser/blob/main/example_workflows/example1.png)


# 5 Tips

* 对于水印，mask_dilation_iter（遮罩膨胀系数）应适当调低，比如2，常规使用Propainter的采样也就够了（最大边是960，应用了再超分吧） 
* 现在12G也能跑1280*720，DiffuEraser的sample 节点的 blend支持2种输出，关闭为降低闪烁，开启为使用合成，避免loop循环的反复加载模型
* cfg>1.0 时 negative 生效


# 6 Citation
```
@misc{li2025diffueraserdiffusionmodelvideo,
   title={DiffuEraser: A Diffusion Model for Video Inpainting},
   author={Xiaowen Li and Haolan Xue and Peiran Ren and Liefeng Bo},
   year={2025},
   eprint={2501.10018},
   archivePrefix={arXiv},
   primaryClass={cs.CV},
   url={https://arxiv.org/abs/2501.10018},
}
```
```
@inproceedings{zhou2023propainter,
   title={{ProPainter}: Improving Propagation and Transformer for Video Inpainting},
   author={Zhou, Shangchen and Li, Chongyi and Chan, Kelvin C.K and Loy, Chen Change},
   booktitle={Proceedings of IEEE International Conference on Computer Vision (ICCV)},
   year={2023}
}
```
```
@misc{ju2024brushnet,
  title={BrushNet: A Plug-and-Play Image Inpainting Model with Decomposed Dual-Branch Diffusion},
  author={Xuan Ju and Xian Liu and Xintao Wang and Yuxuan Bian and Ying Shan and Qiang Xu},
  year={2024},
  eprint={2403.06976},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```
```
@article{BiRefNet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  year={2024}
}
```
