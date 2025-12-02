import gc
import copy
import cv2
import os
import numpy as np
import torch
import torchvision
import re
import random
from einops import repeat
from PIL import Image, ImageFilter
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UniPCMultistepScheduler,
    LCMScheduler,
    StableDiffusionPipeline
)
from diffusers.schedulers import TCDScheduler
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.utils.torch_utils import randn_tensor
from transformers import AutoTokenizer, PretrainedConfig
from safetensors.torch import load_file
from .unet_motion_model import MotionAdapter, UNetMotionModel
from .brushnet_CA import BrushNetModel
from .unet_2d_condition import UNet2DConditionModel
from .pipeline_diffueraser import StableDiffusionDiffuEraserPipeline


def extract_step_number(ckpt_name):
    # 使用正则表达式查找 "step" 前面的数字
    match = re.search(r'(\d+)-Step', ckpt_name)
    if match:
        return int(match.group(1))
    else:
        return 2

checkpoints = {
    "2-Step": ["pcm_{}_smallcfg_2step_converted.safetensors", 2, 0.0],
    "4-Step": ["pcm_{}_smallcfg_4step_converted.safetensors", 4, 0.0],
    "8-Step": ["pcm_{}_smallcfg_8step_converted.safetensors", 8, 0.0],
    "16-Step": ["pcm_{}_smallcfg_16step_converted.safetensors", 16, 0.0],
    "Normal CFG 4-Step": ["pcm_{}_normalcfg_4step_converted.safetensors", 4, 7.5],
    "Normal CFG 8-Step": ["pcm_{}_normalcfg_8step_converted.safetensors", 8, 7.5],
    "Normal CFG 16-Step": ["pcm_{}_normalcfg_16step_converted.safetensors", 16, 7.5],
    "LCM-Like LoRA": [
        "pcm_{}_lcmlike_lora_converted.safetensors",
        4,
        0.0,
    ],
}

def import_model_class_from_model_name_or_path(pretrained_model_name_or_path: str, revision: str):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        revision=revision,
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "RobertaSeriesModelWithTransformation":
        try:    
            from diffusers.pipelines.alt_diffusion.modeling_roberta_series import RobertaSeriesModelWithTransformation # old diffusers version
            return RobertaSeriesModelWithTransformation
        except:
            print("Error: Could not import RobertaSeriesModelWithTransformation.")
            raise ValueError(f"{model_class} is not supported.")
       
    else:
        raise ValueError(f"{model_class} is not supported.")

def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        frames = [f.resize(process_size) for f in frames]  
    else:
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]  
    
    return frames

def read_mask(validation_mask, fps, n_total_frames, img_size, mask_dilation_iter, frames):
    cap = cv2.VideoCapture(validation_mask)
    if not cap.isOpened():
        print("Error: Could not open mask video.")
        exit()
    mask_fps = cap.get(cv2.CAP_PROP_FPS)
    if mask_fps != fps:
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    masks = []
    masked_images = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:  
            break
        if(idx >= n_total_frames):
            break
        mask = Image.fromarray(frame[...,::-1]).convert('L')
        if mask.size != img_size:
            mask = mask.resize(img_size, Image.NEAREST)
        mask = np.asarray(mask)
        m = np.array(mask > 0).astype(np.uint8)
        m = cv2.erode(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=1)
        m = cv2.dilate(m,
                    cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                    iterations=mask_dilation_iter)

        mask = Image.fromarray(m * 255)
        masks.append(mask)

        masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
        masked_image = Image.fromarray(masked_image.astype(np.uint8))
        masked_images.append(masked_image)

        idx += 1
    cap.release()

    return masks, masked_images

def read_priori(priori, fps, n_total_frames, img_size):
    cap = cv2.VideoCapture(priori)
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    priori_fps = cap.get(cv2.CAP_PROP_FPS)
   
    if (priori_fps - fps) > 1e-8:
        print(f"priori fps: {priori_fps}, fps: {fps}")
        cap.release()
        raise ValueError("The frame rate of all input videos needs to be consistent.")

    prioris=[]
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break
        if(idx >= n_total_frames):
            break
        img = Image.fromarray(frame[...,::-1])
        if img.size != img_size:
            img = img.resize(img_size)
        prioris.append(img)
        idx += 1
    cap.release()

    os.remove(priori) # remove priori 

    return prioris

def read_video(validation_image, video_length, nframes, max_img_size):
    vframes, aframes, info = torchvision.io.read_video(filename=validation_image, pts_unit='sec', end_pts=video_length) # RGB
    fps = info['video_fps']
    n_total_frames = int(video_length * fps)
    n_clip = int(np.ceil(n_total_frames/nframes))

    frames = list(vframes.numpy())[:n_total_frames]
    frames = [Image.fromarray(f) for f in frames]
    max_size = max(frames[0].size)
    if(max_size<256):
        raise ValueError("The resolution of the uploaded video must be larger than 256x256.")
    if(max_size>4096):
        raise ValueError("The resolution of the uploaded video must be smaller than 4096x4096.")
    if max_size>max_img_size:
        ratio = max_size/max_img_size
        ratio_size = (int(frames[0].size[0]/ratio),int(frames[0].size[1]/ratio))
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    elif (frames[0].size[0]%8==0) and (frames[0].size[1]%8==0):
        img_size = frames[0].size
        resize_flag=False
    else:
        ratio_size = frames[0].size
        img_size = (ratio_size[0]-ratio_size[0]%8, ratio_size[1]-ratio_size[1]%8)
        resize_flag=True
    if resize_flag:
        frames = resize_frames(frames, img_size)
        img_size = frames[0].size

    return frames, fps, img_size, n_clip, n_total_frames


class DiffuEraser:
    def __init__(self, device, ):
        self.device = device

    def load_model(self,repo, diffueraser_path, ckpt_path,original_config_file,ckpt="Normal CFG 4-Step",):
        self.noise_scheduler = DDPMScheduler.from_pretrained(repo, 
                subfolder="scheduler",
                prediction_type="v_prediction",
                timestep_spacing="trailing",
                rescale_betas_zero_snr=True
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
                    repo,
                    subfolder="tokenizer",
                    use_fast=False,
                )
        vae_config=AutoencoderKL.load_config(os.path.join(repo,"vae/config.json"))
        self.vae=AutoencoderKL.from_config(vae_config)
        self.vae.load_state_dict(load_file(ckpt_path) if ckpt_path.endswith(".safetensors") else torch.load(ckpt_path,weights_only=False),strict=False)
        #self.vae=AutoencoderKL.from_single_file(ckpt_path,config=os.path.join(repo,"vae") )
        # try: 
        #     pipe = StableDiffusionPipeline.from_single_file(
        #     ckpt_path,config=repo, original_config=original_config_file)
        # except:
        #     pipe = StableDiffusionPipeline.from_single_file(
        #     ckpt_path, config=repo,original_config_file=original_config_file)

        # self.text_encoder = pipe.text_encoder
        #self.vae = pipe.vae
        #del pipe 
        gc.collect()
        torch.cuda.empty_cache()

        self.brushnet = BrushNetModel.from_pretrained(diffueraser_path, subfolder="brushnet")
        self.unet_main = UNetMotionModel.from_pretrained(
            diffueraser_path, subfolder="unet_main",
        )
        ## set pipeline
        self.pipeline = StableDiffusionDiffuEraserPipeline.from_pretrained(
            repo,
            vae=self.vae,
            text_encoder=None,
            tokenizer=self.tokenizer,
            unet=self.unet_main,
            brushnet=self.brushnet,
            safety_checker=None,#no need 
        ).to(self.device, torch.float16)
        # self.vae=None
        # self.text_encoder=None
        self.pipeline.scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.pipeline.set_progress_bar_config(disable=True)

        self.noise_scheduler = UniPCMultistepScheduler.from_config(self.pipeline.scheduler.config)
        self.vae_scale_factor = 2 ** (len(self.pipeline.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True)
        
        try:
            self.pipeline.load_lora_weights(pretrained_model_name_or_path_or_dict=ckpt)
            print("Loaded lora from", ckpt)
        except Exception as e:
            print(f"Failed to apply LoRA {str(e)}")
            pass
        
        if "lcmlike" in ckpt.lower():
            self.pipeline.scheduler = LCMScheduler()
            self.num_inference_steps= 4
        else:
            self.pipeline.scheduler = TCDScheduler(
                    num_train_timesteps=1000,
                    beta_start=0.00085,
                    beta_end=0.012,
                    beta_schedule="scaled_linear",
                    timestep_spacing="trailing",
                )
            self.num_inference_steps=extract_step_number(ckpt)


        #self.num_inference_steps = checkpoints[ckpt][1]
        
        if "normal" in ckpt.lower():
            self.guidance_scale = 7.5
        else:
            self.guidance_scale = 0
        #self.guidance_scale = 0


    def to(self, device):
        self.device=device
        self.pipeline.to(device)

    def forward(self, validation_image, validation_mask, prioris, output_path,positive,load_videobypath=False,
                max_img_size = 1280, video_length=2, mask_dilation_iter=4,
                nframes=22, seed=None, revision = None, guidance_scale=None, blended=True,num_inference_steps=None,fps=24,img_size=(512, 512),if_save_video=False):
        validation_prompt = ""  # 
        guidance_scale_final = self.guidance_scale if guidance_scale==None else guidance_scale
        num_inference_steps_final = self.num_inference_steps if num_inference_steps==None else num_inference_steps

        if (max_img_size<256 or max_img_size>1920):
            raise ValueError("The max_img_size must be larger than 256, smaller than 1920.")

        ################ read input video ################ 
        if load_videobypath:
            frames, fps, img_size, n_clip, n_total_frames = read_video(validation_image, video_length, nframes, max_img_size)
        else:
            frames=validation_image
            n_total_frames=len(validation_image)
            n_clip = int(np.ceil(n_total_frames/nframes))
        video_len = len(frames)
        #frames[0].save("input0.png")
        ################     read mask    ################ 
        if load_videobypath:
            validation_masks_input, validation_images_input = read_mask(validation_mask, fps, video_len, img_size, mask_dilation_iter, frames)
        else:
            validation_masks_list=[i.convert('L') for i in validation_mask.copy()]
            validation_images_input=[]
            validation_masks_input=[]
            for idx ,mask in enumerate(validation_masks_list):
                #mask = Image.fromarray(i[...,::-1]).convert('L')
                if mask.size != img_size:
                    mask = mask.resize(img_size, Image.NEAREST)
                mask = np.asarray(mask)
                m = np.array(mask > 0).astype(np.uint8)
                m = cv2.erode(m,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=1)
                m = cv2.dilate(m,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                            iterations=mask_dilation_iter)

                mask = Image.fromarray(m * 255)
                validation_masks_input.append(mask)
                masked_image = np.array(frames[idx])*(1-(np.array(mask)[:,:,np.newaxis].astype(np.float32)/255))
                masked_image = Image.fromarray(masked_image.astype(np.uint8))
                validation_images_input.append(masked_image)
       
        ################    read priori   ################  
        #validation_images_input[0].save("input1.png")
        #prioris = read_priori(priori, fps, n_total_frames, img_size)
        if prioris[0].size != img_size:
            prioris = [img.resize(img_size) for img in prioris]
        ## recheck
        n_total_frames = min(min(len(frames), len(validation_masks_input)), len(prioris))
        if(n_total_frames<22):
            raise ValueError("The effective video duration is too short. Please make sure that the number of frames of video, mask, and priori is at least greater than 22 frames.")
        validation_masks_input = validation_masks_input[:n_total_frames]
        validation_images_input = validation_images_input[:n_total_frames]
        frames = frames[:n_total_frames]
        prioris = prioris[:n_total_frames]

        prioris = resize_frames(prioris)
        validation_masks_input = resize_frames(validation_masks_input)
        validation_images_input = resize_frames(validation_images_input)
        resized_frames = resize_frames(frames)
        #resized_frames[0].save("input2.png")

        ##############################################
        # DiffuEraser inference
        ##############################################
        print("DiffuEraser inference...")
        if seed is None:
            generator = None
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        ## random noise
        real_video_length = len(validation_images_input)
        tar_width, tar_height = validation_images_input[0].size 
        shape = (
            nframes,
            4,
            tar_height//8,
            tar_width//8
        )

        if self.unet_main is not None:
            prompt_embeds_dtype = self.unet_main.dtype
        else:
            prompt_embeds_dtype = torch.float16
        noise_pre = randn_tensor(shape, device=torch.device(self.device), dtype=prompt_embeds_dtype, generator=generator) 
        noise = repeat(noise_pre, "t c h w->(repeat t) c h w", repeat=n_clip)[:real_video_length,...]
        
        ################  prepare priori  ################
        images_preprocessed = []
        for image in prioris:
            image = self.image_processor.preprocess(image, height=tar_height, width=tar_width).to(dtype=torch.float32)
            image = image.to(device=torch.device(self.device), dtype=torch.float16)
            images_preprocessed.append(image)
        pixel_values = torch.cat(images_preprocessed)

        with torch.no_grad():
            pixel_values = pixel_values.to(dtype=torch.float16)
            latents = []
            num=4
            for i in range(0, pixel_values.shape[0], num):
                latents.append(self.pipeline.vae.encode(pixel_values[i : i + num]).latent_dist.sample())      
            latents = torch.cat(latents, dim=0)  
        latents = latents * self.pipeline.vae.config.scaling_factor #[(b f), c1, h, w], c1=4
        self.pipeline.vae.to("cpu")
        torch.cuda.empty_cache()  
        timesteps = torch.tensor([0], device=self.device)
        timesteps = timesteps.long()

        validation_masks_input_ori = copy.deepcopy(validation_masks_input)
        resized_frames_ori = copy.deepcopy(resized_frames)

        ################  Pre-inference  ################
        if n_total_frames > nframes*2: ## do pre-inference only when number of input frames is larger than nframes*2
            ## sample
            step = n_total_frames / nframes
            sample_index = [int(i * step) for i in range(nframes)]
            sample_index = sample_index[:22]
            validation_masks_input_pre = [validation_masks_input[i] for i in sample_index]
            validation_images_input_pre = [validation_images_input[i] for i in sample_index]
            latents_pre = torch.stack([latents[i] for i in sample_index])

            ## add proiri
            noisy_latents_pre = self.noise_scheduler.add_noise(latents_pre, noise_pre, timesteps) 
            latents_pre = noisy_latents_pre

            with torch.no_grad():
                latents_pre_out = self.pipeline(
                    num_frames=nframes, 
                    prompt=None, 
                    images=validation_images_input_pre, 
                    masks=validation_masks_input_pre, 
                    prompt_embeds=positive[0][0], 
                    num_inference_steps=num_inference_steps_final, 
                    generator=generator,
                    guidance_scale=guidance_scale_final,
                    latents=latents_pre,
                ).latents
            torch.cuda.empty_cache()  

            def decode_latents(latents, weight_dtype):
                latents = 1 / self.pipeline.vae.config.scaling_factor * latents
                video = []
                for t in range(latents.shape[0]):
                    video.append(self.pipeline.vae.decode(latents[t:t+1, ...].to(weight_dtype)).sample)
                video = torch.concat(video, dim=0)
                # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
                video = video.float()
                return video
            with torch.no_grad():
                video_tensor_temp = decode_latents(latents_pre_out, weight_dtype=torch.float16)
                images_pre_out  = self.image_processor.postprocess(video_tensor_temp, output_type="pil")
            torch.cuda.empty_cache()  

            ## replace input frames with updated frames
            black_image = Image.new('L', validation_masks_input[0].size, color=0)
            for i,index in enumerate(sample_index):
                latents[index] = latents_pre_out[i]
                validation_masks_input[index] = black_image
                validation_images_input[index] = images_pre_out[i]
                resized_frames[index] = images_pre_out[i]
          
        else:
            latents_pre_out=None
            sample_index=None
        gc.collect()
        torch.cuda.empty_cache()

        ################  Frame-by-frame inference  ################
        ## add priori
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps) 
        latents = noisy_latents
        with torch.no_grad():
            images = self.pipeline(
                num_frames=nframes, 
                prompt=None, 
                images=validation_images_input, 
                masks=validation_masks_input,
                prompt_embeds=positive[0][0], 
                num_inference_steps=num_inference_steps_final, 
                generator=generator,
                guidance_scale=guidance_scale_final,
                latents=latents,
            ).frames
        images = images[:real_video_length]

        gc.collect()
        torch.cuda.empty_cache()

        ################ Compose ################
        binary_masks = validation_masks_input_ori
        mask_blurreds = []
        if blended:
            for i in range(len(binary_masks)):
                mask_array = np.array(binary_masks[i])
                mask_blurred = morphological_edge_blur(np.array(mask_array), sigma=2.0, edge_width=3)       
                #mask_blurred = cv2.GaussianBlur(np.array(binary_masks[i]), blur_kernel, 0)/255.
                binary_mask = 1-(1-mask_array/255.) * (1-mask_blurred)
                mask_blurreds.append(Image.fromarray((binary_mask*255).astype(np.uint8)))
            binary_masks = mask_blurreds

            comp_frames = []
            for i in range(len(images)):
                mask = np.expand_dims(np.array(binary_masks[i]),2).repeat(3, axis=2).astype(np.float32)/255.
                img = (np.array(images[i]).astype(np.uint8) * mask + np.array(resized_frames_ori[i]).astype(np.uint8) * (1 - mask)).astype(np.uint8)
                comp_frames.append(Image.fromarray(img))
        else:
            comp_frames = simple_flicker_smoothing(images, alpha=0.15)

        if if_save_video:
            default_fps = fps
            prefix = ''.join(random.choice("0123456789") for _ in range(6))
            priori_path = os.path.join(output_path, f"priori_{prefix}.mp4")        
            os.makedirs(os.path.dirname(priori_path), exist_ok=True)
            
            writer = cv2.VideoWriter(priori_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                default_fps, comp_frames[0].size)
            for f in range(real_video_length):
                img = np.array(comp_frames[f]).astype(np.uint8)
                writer.write(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            writer.release()
        ################################

        return comp_frames
def simple_flicker_smoothing(frames, alpha=0.1):
    """
    简单的闪烁平滑，最小化对动态内容的影响
    """
    if len(frames) < 2:
        return frames
    
    smoothed_frames = [frames[0]]
    
    for i in range(1, len(frames)):
        current = np.array(frames[i]).astype(np.float32)
        previous = np.array(frames[i-1]).astype(np.float32)
        
        # 只对变化很小的像素进行平滑（可能是闪烁）
        diff = np.abs(current - previous)
        static_mask = (diff < 10.0).astype(np.float32)  # 阈值可根据需要调整
        
        # 只在静态区域应用轻微平滑
        smoothed = previous * alpha * static_mask + current * (1 - alpha * static_mask)
        
        smoothed_frames.append(Image.fromarray(np.clip(smoothed, 0, 255).astype(np.uint8)))
    
    return smoothed_frames

def morphological_edge_blur(mask, sigma=3.0, edge_width=5):
    """
    使用形态学操作提取边缘并只模糊边缘区域
    """
    if mask.dtype != np.float32:
        mask_float = mask.astype(np.float32)
    else:
        mask_float = mask.copy()
    
    # 转换为二值图像
    binary_mask = (mask_float > 0.5).astype(np.uint8)
    
    if not np.any(binary_mask):
        return mask_float
    
    # 创建边缘遮罩
    # 腐蚀操作缩小遮罩
    kernel_size = max(3, edge_width)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    
    # 边缘 = 原始遮罩 - 腐蚀后的遮罩
    edge_mask = binary_mask - eroded
    
    # 只对边缘区域进行高斯模糊
    edge_region = mask_float * edge_mask.astype(np.float32)
    
    # 模糊边缘区域
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    ksize = max(3, min(101, ksize if ksize % 2 == 1 else ksize + 1))
    blurred_edges = cv2.GaussianBlur(edge_region, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
    
    # 合成结果：内部保持原值，边缘使用模糊值
    inner_region = mask_float * eroded.astype(np.float32)
    result = inner_region + blurred_edges
    
    return np.clip(result, 0.0, 1.0)
