# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
import gc
import numpy as np
from .node_utils import  load_images,tensor2pil_list,file_exists,download_weights,image2masks
import folder_paths
from .run_diffueraser import load_diffueraser,diffueraser_inference


MAX_SEED = np.iinfo(np.int32).max
current_node_path = os.path.dirname(os.path.abspath(__file__))
device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

# add checkpoints dir
DiffuEraser_weigths_path = os.path.join(folder_paths.models_dir, "DiffuEraser")
if not os.path.exists(DiffuEraser_weigths_path):
    os.makedirs(DiffuEraser_weigths_path)
folder_paths.add_model_folder_path("DiffuEraser", DiffuEraser_weigths_path)



class DiffuEraserLoader:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "checkpoint": (["none"] + folder_paths.get_filename_list("checkpoints"),),
                "lora": (["none"] + folder_paths.get_filename_list("loras"),),
            },
        }

    RETURN_TYPES = ("MODEL_DiffuEraser",)
    RETURN_NAMES = ("model",)
    FUNCTION = "loader_main"
    CATEGORY = "DiffuEraser"

    def loader_main(self,checkpoint,lora):

        # check model is exits or not,if not auto downlaod

        brushnet_weigths_path = os.path.join(DiffuEraser_weigths_path, "brushnet")
        if not os.path.exists(brushnet_weigths_path):
            os.makedirs(brushnet_weigths_path)

        unet_weigths_path = os.path.join(DiffuEraser_weigths_path, "unet_main")    
        if not os.path.exists(unet_weigths_path):
            os.makedirs(unet_weigths_path)


        if not file_exists(brushnet_weigths_path,"config.json") :
            download_weights(DiffuEraser_weigths_path,"lixiaowen/diffuEraser",subfolder="brushnet",pt_name="config.json")
        if not file_exists(brushnet_weigths_path,"diffusion_pytorch_model.safetensors"):
            download_weights(DiffuEraser_weigths_path,"lixiaowen/diffuEraser",subfolder="brushnet",pt_name="diffusion_pytorch_model.safetensors")

        if not file_exists(unet_weigths_path,"diffusion_pytorch_model.safetensors"):
            download_weights(DiffuEraser_weigths_path,"lixiaowen/diffuEraser",subfolder="unet_main",pt_name="diffusion_pytorch_model.safetensors")
        if not file_exists(unet_weigths_path,"config.json"):
            download_weights(DiffuEraser_weigths_path,"lixiaowen/diffuEraser",subfolder="unet_main",pt_name="config.json")

        # load model
        original_config_file=os.path.join(current_node_path,"libs/v1-inference.yaml")
        sd_repo=os.path.join(current_node_path,"sd15_repo")
        if checkpoint!="none":
            ckpt_path=folder_paths.get_full_path("checkpoints",checkpoint)
        else:
            raise "no sd1.5 checkpoint"
        
        if lora!="none":
            pcm_lora_path=folder_paths.get_full_path("loras",lora)
        else:
            raise "no pcm lora checkpoint"
        # if vae!="none" :    
        #     vae_path=folder_paths.get_full_path("vae",vae)
        # else:
        #     raise "no sd1.5 vae"


        model=load_diffueraser(DiffuEraser_weigths_path, pcm_lora_path,sd_repo,ckpt_path,original_config_file,device)

        gc.collect()
        torch.cuda.empty_cache()
        return (model,)
    
class DiffuEraserSampler:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL_DiffuEraser",),
                "images": ("IMAGE",), #[b,h,w,c]
                "fps": ("FLOAT", {"forceInput": True,}),
                "seed": ("INT", {"default": -1, "min": -1, "max": MAX_SEED}),
                "num_inference_steps": ("INT", {
                    "default": 2,
                    "min": 1,  # Minimum value
                    "max": 120,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "guidance_scale": ("FLOAT", {"default": 0, "min": 0, "max": 10., "step": -0.1, "display": "number"}),
                "video_length": ("INT", {
                    "default": 10,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "mask_dilation_iter": ("INT", {
                    "default": 8,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "ref_stride": ("INT", {
                    "default": 10,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "neighbor_length": ("INT", {
                    "default": 10,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "subvideo_length": ("INT", {
                    "default": 50,
                    "min": 1,  # Minimum value
                    "max": 1024,  # Maximum value
                    "step": 1,  # Slider's step
                    "display": "number",  # Cosmetic only: display as "number" or "slider"
                }),
                "video2mask":("BOOLEAN", {"default": False},),
                "seg_repo": ("STRING", {"default": "briaai/RMBG-2.0"},),
                "save_result_video":("BOOLEAN", {"default": False},),},
                "optional": {
                    "video_mask": ("IMAGE",),
                  
                }
                         
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE","STRING", )
    RETURN_NAMES = ("images","propainter_img","output_path", )
    FUNCTION = "sampler_main"
    CATEGORY = "DiffuEraser"
    
    def sampler_main(self, model,images,fps,seed,num_inference_steps,guidance_scale,video_length,mask_dilation_iter,ref_stride,neighbor_length,subvideo_length,video2mask,seg_repo,save_result_video,**kwargs):
        
        video_inpainting_sd=model.get("video_inpainting_sd")
        propainter=model.get("propainter")

        max_img_size=1920
        _,height,width,_  = images.size()
        video_image=tensor2pil_list(images,width,height)
        if video2mask and seg_repo:   
            print("***********Start video to masks infer **************")
            video_mask=image2masks(seg_repo,video_image)# use rmbg or BiRefNet to make video to masks
        else:
            if isinstance(kwargs.get("video_mask"),torch.Tensor):
                video_mask=tensor2pil_list(kwargs.get("video_mask"),width,height)
            else:
                raise "no video_mask,you can enable video2mask and fill a rmbg or BiRefNet repo to generate mask from video_image,or link video_mask from other node"

        seeds=None if seed==-1 else seed

        print("frame_length:",len(video_image),"mask_length:",len(video_mask),"fps:",fps)
        if len(video_mask)!=len(video_image) :
            if  len(video_mask)==1:
                video_mask=video_mask*len(video_image) # if use one mask to inpaint all frames
            else:
                if len(video_mask)>len(video_image):  
                    video_mask=video_mask[:len(video_image)]
                    print("video_mask length:",len(video_mask),"video_image length:",len(video_image))
                else:
                    video_mask=video_mask+video_mask[:len(video_image)-len(video_mask)]
                    print("video_mask length:",len(video_mask),"video_image length:",len(video_image))
                
      
        
        print("***********Start DiffuEraser Sampler**************")
        video_inpainting_sd.to(device)
        propainter.to(device)
        output_path,image_list,Propainter_list=diffueraser_inference(video_inpainting_sd,propainter,video_image,video_mask,video_length,width,height,
                                          mask_dilation_iter,max_img_size,ref_stride,neighbor_length,subvideo_length,guidance_scale,num_inference_steps,seeds,fps,save_result_video)
        video_inpainting_sd.to("cpu")
        #propainter.to("cpu")

        images=load_images(image_list)
        Propainter_img=load_images(Propainter_list)
        gc.collect()
        torch.cuda.empty_cache()
        return (images,Propainter_img,output_path,)



NODE_CLASS_MAPPINGS = {
    "DiffuEraserLoader":DiffuEraserLoader,
    "DiffuEraserSampler":DiffuEraserSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DiffuEraserLoader":"DiffuEraserLoader",
    "DiffuEraserSampler":"DiffuEraserSampler",
}

