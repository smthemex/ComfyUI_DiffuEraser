import torch
import os 
import time
import random
from .libs.diffueraser import DiffuEraser
from .propainter.inference import Propainter, get_device
import folder_paths
import gc


def load_diffueraser(pre_model_path, pcm_lora_path,vae_path,sd_repo,ckpt_path,original_config_file,device):

    start_time = time.time()
    device = get_device()
    #ckpt = "2-Step"
    propainter_model_dir=os.path.join(pre_model_path, "propainter")
    if not os.path.exists(propainter_model_dir):
        os.makedirs(propainter_model_dir)
    video_inpainting_sd = DiffuEraser(device, sd_repo, vae_path, pre_model_path,ckpt_path,original_config_file, ckpt=pcm_lora_path)
    propainter = Propainter(propainter_model_dir, device=device)

    end_time = time.time()  
    load_time = end_time - start_time  
    print(f"DiffuEraser load time: {load_time:.4f} s")
    return {"video_inpainting_sd":video_inpainting_sd,"propainter":propainter}


def diffueraser_inference(video_inpainting_sd,propainter,input_video,input_mask,video_length,width,height,mask_dilation_iter,
                          max_img_size,ref_stride,neighbor_length,subvideo_length,guidance_scale,num_inference_steps,seed,fps,save_result_video,):

    prefix = ''.join(random.choice("0123456789") for _ in range(6))
    priori_path = os.path.join(folder_paths.get_output_directory(), f"priori_{prefix}.mp4")        
    if not os.path.exists(os.path.dirname(priori_path)):
        os.makedirs(os.path.dirname(priori_path))                
    output_path = os.path.join(folder_paths.get_output_directory(), f"diffueraser_result_{prefix}.mp4") 
    start_time = time.time()
    load_videobypath=False
    # if load_videobypath:
    #     input_mask="F:/test/ComfyUI/input/mask.mp4"
    #     input_video="F:/test/ComfyUI/input/video.mp4"
    ## priori
    res=propainter.forward(input_video, input_mask, priori_path,load_videobypath=load_videobypath,video_length=video_length, height=height,width=width,
                        ref_stride=ref_stride, neighbor_length=neighbor_length, subvideo_length = subvideo_length,
                        mask_dilation = mask_dilation_iter,save_fps=fps) 

    propainter.to("cpu")
    gc.collect()
    torch.cuda.empty_cache()
    ## diffueraser
    # The default value is 0.  
    video_path,image_list=video_inpainting_sd.forward(input_video, input_mask, priori_path,output_path,load_videobypath=load_videobypath,
                                max_img_size = max_img_size, video_length=video_length, mask_dilation_iter=mask_dilation_iter,seed=seed,
                                guidance_scale=guidance_scale,num_inference_steps=num_inference_steps,fps=fps,img_size=(width,height),if_save_video=save_result_video)
    
    end_time = time.time()  
    inference_time = end_time - start_time  
    print(f"DiffuEraser inference time: {inference_time:.4f} s")

    torch.cuda.empty_cache()
    return video_path,image_list

