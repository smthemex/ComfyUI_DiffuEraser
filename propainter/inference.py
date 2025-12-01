# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm
import torch
import torchvision
import gc

# try:
#     from model.modules.flow_comp_raft import RAFT_bi
#     from model.recurrent_flow_completion import RecurrentFlowCompleteNet
#     from model.propainter import InpaintGenerator
#     from utils.download_util import load_file_from_url
#     from core.utils import to_tensors
#     from model.misc import get_device
# except:
from .model.modules.flow_comp_raft import RAFT_bi
from .model.recurrent_flow_completion import RecurrentFlowCompleteNet
from .model.propainter import InpaintGenerator
from .utils.download_util import load_file_from_url
from .core.utils import to_tensors
from .model.misc import get_device

import warnings
warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'
MaxSideThresh = 960


# resize frames
def resize_frames(frames, size=None):    
    if size is not None:
        out_size = size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]
            
    else:
        out_size = frames[0].size
        process_size = (out_size[0]-out_size[0]%8, out_size[1]-out_size[1]%8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]       
    return frames, process_size, out_size

#  read frames from video
def read_frame_from_videos(frame_root, video_length):
    if frame_root.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        video_name = os.path.basename(frame_root)[:-4]
        vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec', end_pts=video_length) # RGB
        frames = list(vframes.numpy())
        frames = [Image.fromarray(f) for f in frames]
        fps = info['video_fps']
        nframes = len(frames)
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))
        for fr in fr_lst:
            frame = cv2.imread(os.path.join(frame_root, fr))
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None
        nframes = len(frames)
    size = frames[0].size

    return frames, fps, size, video_name, nframes

def binary_mask(mask, th=0.1):
    mask[mask>th] = 1
    mask[mask<=th] = 0
    return mask
  
# read frame-wise masks
def read_mask(mpath, frames_len, size, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []
    
    if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')): # input single img path
        masks_img = [Image.open(mpath)]
    elif mpath.endswith(('mp4', 'mov', 'avi', 'MP4', 'MOV', 'AVI')): # input video path
        cap = cv2.VideoCapture(mpath)
        if not cap.isOpened():
            print("Error: Could not open video.")
            exit()
        idx = 0
        while True:
            ret, frame = cap.read()
            if not ret: 
                break
            if(idx >= frames_len):
                break
            masks_img.append(Image.fromarray(frame))
            idx += 1
        cap.release()
    else:  
        mnames = sorted(os.listdir(mpath))
        for mp in mnames:
            masks_img.append(Image.open(os.path.join(mpath, mp)))
            # print(mp)
          
    for mask_img in masks_img:
        if size is not None:
            mask_img = mask_img.resize(size, Image.NEAREST)
        mask_img = np.array(mask_img.convert('L'))

        # Dilate 8 pixel so that all known pixel is trustworthy
        if flow_mask_dilates > 0:
            flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask_img = binary_mask(mask_img).astype(np.uint8)
        # Close the small holes inside the foreground objects
        # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
        # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask_img * 255))
        
        if mask_dilates > 0:
            mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
        else:
            mask_img = binary_mask(mask_img).astype(np.uint8)
        masks_dilated.append(Image.fromarray(mask_img * 255))
    
    if len(masks_img) == 1:
        flow_masks = flow_masks * frames_len
        masks_dilated = masks_dilated * frames_len

    return flow_masks, masks_dilated

def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=10, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids:
                ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num:
                    break
                ref_index.append(i)
    return ref_index

def file_exists(directory, filename):
    # 构建文件的完整路径
    file_path = os.path.join(directory, filename)
    # 检查文件是否存在
    return os.path.isfile(file_path)

class Propainter:
    def __init__(self, device):
        self.device = device

    def load_propainter(self,fix_raft_path,flow_path,ProPainter_path):
        ##############################################
        # set up RAFT and flow competition model
        ##############################################

        # if file_exists(propainter_model_dir,'raft-things.pth'):
        #     ckpt_path=os.path.join(propainter_model_dir,'raft-things.pth')
        # else:
        #     ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'), 
        #                                 model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_raft = RAFT_bi(fix_raft_path, self.device)
        # if file_exists(propainter_model_dir,'recurrent_flow_completion.pth'):
        #     ckpt_path_=os.path.join(propainter_model_dir,'recurrent_flow_completion.pth')
        # else:
        #     ckpt_path_ = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'), 
        #                                 model_dir=propainter_model_dir, progress=True, file_name=None)
        self.fix_flow_complete = RecurrentFlowCompleteNet(flow_path)
        for p in self.fix_flow_complete.parameters():
            p.requires_grad = False
        self.fix_flow_complete.to(self.device)
        self.fix_flow_complete.eval()

        ##############################################
        # set up ProPainter model
        ##############################################

        # if file_exists(propainter_model_dir,'ProPainter.pth'):
        #     ckpt_path_p=os.path.join(propainter_model_dir,'ProPainter.pth')
        # else:
        #     ckpt_path_p = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'), 
        #                                 model_dir=propainter_model_dir, progress=True, file_name=None)
        self.model = InpaintGenerator(model_path=ProPainter_path).to(self.device)
        self.model.eval()

    def to(self, device):
        self.device = device
        self.fix_raft.to(device)
        self.fix_flow_complete.to(device)
        self.model.to(device)

    def forward(self, video, mask,load_videobypath=False, resize_ratio=1.0, video_length=2, height=-1, width=-1,
                mask_dilation=4, ref_stride=10, neighbor_length=10, subvideo_length=80,
                raft_iter=20, save_fps=24.0, fp16=True):
        
        # Use fp16 precision during inference to reduce running memory cost
        use_half = True if fp16 else False 
        if self.device == torch.device('cpu'):
            use_half = False

        ################ read input video ################ 
        if load_videobypath:
            frames, fps, size, video_name, nframes = read_frame_from_videos(video, video_length)
        else:
            frames=video
            size = (width, height)
            video_name='input'
            nframes=len(frames)
            fps=None
    
            
        #frames, fps, size, video_name, nframes = read_frame_from_videos(video, video_length)
        frames = frames[:nframes]
        if not width == -1 and not height == -1:
            size = (width, height)

        longer_edge = max(size[0], size[1])
        if(longer_edge > MaxSideThresh): 
            print('input video size is too large, resize to', MaxSideThresh)
            scale = MaxSideThresh / longer_edge
            resize_ratio = resize_ratio * scale
        if  resize_ratio < 1.0: # if longer_edge>960 resize to 960
            origin_size = (int(resize_ratio * size[0]), int(resize_ratio * size[1]))
            frames, size, out_size = resize_frames(frames, origin_size) 
            mask,size_,out_size_=resize_frames(mask, origin_size)
        else:
            origin_size = size
            frames, size, out_size = resize_frames(frames, origin_size)
            mask,size_,out_size_ = resize_frames(mask, origin_size)

        origin_frames=frames.copy()
        origin_masks=mask.copy()

        fps = save_fps if fps is None else fps

        ################ read mask ################ 
        frames_len = len(frames)
        if load_videobypath:
            flow_masks, masks_dilated = read_mask(mask, frames_len, size, 
                                            flow_mask_dilates=mask_dilation,
                                            mask_dilates=mask_dilation)
        else:
            #flow_masks_list=[i.convert('L') for i in mask.copy()]
            masks_dilated = []
            flow_masks = []
            flow_mask_dilates=mask_dilation
            mask_dilates=mask_dilation
            for i in mask.copy():
                # if size is not None: # don't resize mask
                #     i = i.resize(size, Image.NEAREST)
                mask_img = np.array(i.convert('L'))

                # Dilate 8 pixel so that all known pixel is trustworthy
                if flow_mask_dilates > 0:
                    flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates).astype(np.uint8)
                else:
                    flow_mask_img = binary_mask(mask_img).astype(np.uint8)
                # Close the small holes inside the foreground objects
                # flow_mask_img = cv2.morphologyEx(flow_mask_img, cv2.MORPH_CLOSE, np.ones((21, 21),np.uint8)).astype(bool)
                # flow_mask_img = scipy.ndimage.binary_fill_holes(flow_mask_img).astype(np.uint8)
                flow_masks.append(Image.fromarray(flow_mask_img * 255))
                
                if mask_dilates > 0:
                    mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=mask_dilates).astype(np.uint8)
                else:
                    mask_img = binary_mask(mask_img).astype(np.uint8)
                masks_dilated.append(Image.fromarray(mask_img * 255))
            
            if len(mask) == 1:
                flow_masks = flow_masks * frames_len
                masks_dilated = masks_dilated * frames_len
           # masks_dilated = mask.copy()
        flow_masks = flow_masks[:nframes]
        masks_dilated = masks_dilated[:nframes]
        w, h = size

        ################ adjust input ################ 
        frames_len = min(len(frames), len(masks_dilated))
        frames = frames[:frames_len]
        flow_masks = flow_masks[:frames_len]
        masks_dilated = masks_dilated[:frames_len]
        
        ori_frames_inp = [np.array(f).astype(np.uint8) for f in frames]
        frames = to_tensors()(frames).unsqueeze(0) * 2 - 1    
        flow_masks = to_tensors()(flow_masks).unsqueeze(0)
        masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
        frames, flow_masks, masks_dilated = frames.to(self.device), flow_masks.to(self.device), masks_dilated.to(self.device)
 
        ##############################################
        # ProPainter inference
        ##############################################
        video_length = frames.size(1)
        print(f'Priori generating: [{video_length} frames]...')
        with torch.no_grad():
            # ---- compute flow ----
            new_longer_edge = max(frames.size(-1), frames.size(-2))
            if new_longer_edge <= 640: 
                short_clip_len = 12
            elif new_longer_edge <= 720: 
                short_clip_len = 8
            elif new_longer_edge <= 1280:
                short_clip_len = 4
            else:
                short_clip_len = 2

            # use fp32 for RAFT
            if frames.size(1) > short_clip_len:
                gt_flows_f_list, gt_flows_b_list = [], []
                for f in range(0, video_length, short_clip_len):
                    end_f = min(video_length, f + short_clip_len)
                    if f == 0:
                        flows_f, flows_b = self.fix_raft(frames[:,f:end_f], iters=raft_iter)
                    else:
                        flows_f, flows_b = self.fix_raft(frames[:,f-1:end_f], iters=raft_iter)
                    
                    gt_flows_f_list.append(flows_f)
                    gt_flows_b_list.append(flows_b)
                    torch.cuda.empty_cache()
                    
                gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                gt_flows_bi = (gt_flows_f, gt_flows_b)
            else:
                gt_flows_bi = self.fix_raft(frames, iters=raft_iter)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()

            if use_half:
                frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.fix_flow_complete = self.fix_flow_complete.half()
                self.model = self.model.half()
          
            # ---- complete flow ----
            flow_length = gt_flows_bi[0].size(1)
            if flow_length > subvideo_length:
                pred_flows_f, pred_flows_b = [], []
                pad_len = 5
                for f in range(0, flow_length, subvideo_length):
                    s_f = max(0, f - pad_len)
                    e_f = min(flow_length, f + subvideo_length + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(flow_length, f + subvideo_length)
                    pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        flow_masks[:, s_f:e_f+1])
                    pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                        (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), 
                        pred_flows_bi_sub, 
                        flow_masks[:, s_f:e_f+1])

                    pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                    pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()
                    
                pred_flows_f = torch.cat(pred_flows_f, dim=1)
                pred_flows_b = torch.cat(pred_flows_b, dim=1)
                pred_flows_bi = (pred_flows_f, pred_flows_b)
            else:
                pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
                pred_flows_bi = self.fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
                torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            gc.collect()
                

            masks_dilated_ori = masks_dilated.clone()
            # ---- Pre-propagation ----
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if(len(frames[0]))>subvideo_length_img_prop: # perform propagation only when length of frames is larger than subvideo_length_img_prop
                sample_rate = len(frames[0])//(subvideo_length_img_prop//2)
                index_sample =  list(range(0, len(frames[0]), sample_rate))
                sample_frames =  torch.stack([frames[0][i].to(torch.float32) for i in index_sample]).unsqueeze(0) # use fp32 for RAFT
                sample_masks_dilated = torch.stack([masks_dilated[0][i] for i in index_sample]).unsqueeze(0)
                sample_flow_masks =  torch.stack([flow_masks[0][i] for i in index_sample]).unsqueeze(0)
  
                ## recompute flow for sampled frames
                # use fp32 for RAFT
                sample_video_length = sample_frames.size(1)
                if sample_frames.size(1) > short_clip_len:
                    gt_flows_f_list, gt_flows_b_list = [], []
                    for f in range(0, sample_video_length, short_clip_len):
                        end_f = min(sample_video_length, f + short_clip_len)
                        if f == 0:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f:end_f], iters=raft_iter)
                        else:
                            flows_f, flows_b = self.fix_raft(sample_frames[:,f-1:end_f], iters=raft_iter)
                        
                        gt_flows_f_list.append(flows_f)
                        gt_flows_b_list.append(flows_b)
                        torch.cuda.empty_cache()
                        
                    gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
                    gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
                    sample_gt_flows_bi = (gt_flows_f, gt_flows_b)
                else:
                    sample_gt_flows_bi = self.fix_raft(sample_frames, iters=raft_iter)
                    torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()

                if use_half:
                    sample_frames, sample_flow_masks, sample_masks_dilated = sample_frames.half(), sample_flow_masks.half(), sample_masks_dilated.half()
                    sample_gt_flows_bi = (sample_gt_flows_bi[0].half(), sample_gt_flows_bi[1].half())

                # ---- complete flow ----
                flow_length = sample_gt_flows_bi[0].size(1)
                if flow_length > subvideo_length:
                    pred_flows_f, pred_flows_b = [], []
                    pad_len = 5
                    for f in range(0, flow_length, subvideo_length):
                        s_f = max(0, f - pad_len)
                        e_f = min(flow_length, f + subvideo_length + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(flow_length, f + subvideo_length)
                        pred_flows_bi_sub, _ = self.fix_flow_complete.forward_bidirect_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            sample_flow_masks[:, s_f:e_f+1])
                        pred_flows_bi_sub = self.fix_flow_complete.combine_flow(
                            (sample_gt_flows_bi[0][:, s_f:e_f], sample_gt_flows_bi[1][:, s_f:e_f]), 
                            pred_flows_bi_sub, 
                            sample_flow_masks[:, s_f:e_f+1])

                        pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f-s_f-pad_len_e])
                        pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f-s_f-pad_len_e])
                        torch.cuda.empty_cache()
                        
                    pred_flows_f = torch.cat(pred_flows_f, dim=1)
                    pred_flows_b = torch.cat(pred_flows_b, dim=1)
                    sample_pred_flows_bi = (pred_flows_f, pred_flows_b)
                else:
                    sample_pred_flows_bi, _ = self.fix_flow_complete.forward_bidirect_flow(sample_gt_flows_bi, sample_flow_masks)
                    sample_pred_flows_bi = self.fix_flow_complete.combine_flow(sample_gt_flows_bi, sample_pred_flows_bi, sample_flow_masks)
                    torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                gc.collect()
                
                masked_frames = sample_frames * (1 - sample_masks_dilated)
                
                if sample_video_length > subvideo_length_img_prop:
                    updated_frames, updated_masks = [], []
                    pad_len = 10
                    for f in range(0, sample_video_length, subvideo_length_img_prop):
                        s_f = max(0, f - pad_len)
                        e_f = min(sample_video_length, f + subvideo_length_img_prop + pad_len)
                        pad_len_s = max(0, f) - s_f
                        pad_len_e = e_f - min(sample_video_length, f + subvideo_length_img_prop)

                        b, t, _, _, _ = sample_masks_dilated[:, s_f:e_f].size()
                        pred_flows_bi_sub = (sample_pred_flows_bi[0][:, s_f:e_f-1], sample_pred_flows_bi[1][:, s_f:e_f-1])
                        prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                            pred_flows_bi_sub, 
                                                                            sample_masks_dilated[:, s_f:e_f], 
                                                                            'nearest')
                        updated_frames_sub = sample_frames[:, s_f:e_f] * (1 - sample_masks_dilated[:, s_f:e_f]) + \
                                            prop_imgs_sub.view(b, t, 3, h, w) * sample_masks_dilated[:, s_f:e_f]
                        updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                        
                        updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                        torch.cuda.empty_cache()
                        
                    updated_frames = torch.cat(updated_frames, dim=1)
                    updated_masks = torch.cat(updated_masks, dim=1)
                else:
                    b, t, _, _, _ = sample_masks_dilated.size()
                    prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, sample_pred_flows_bi, sample_masks_dilated, 'nearest')
                    updated_frames = sample_frames * (1 - sample_masks_dilated) + prop_imgs.view(b, t, 3, h, w) * sample_masks_dilated
                    updated_masks = updated_local_masks.view(b, t, 1, h, w)
                    torch.cuda.empty_cache()

                ## replace input frames/masks with updated frames/masks 
                for i,index in enumerate(index_sample):
                    frames[0][index] = updated_frames[0][i]
                    masks_dilated[0][index] = updated_masks[0][i]


            # ---- frame-by-frame image propagation ----
            masked_frames = frames * (1 - masks_dilated)
            subvideo_length_img_prop = min(100, subvideo_length) # ensure a minimum of 100 frames for image propagation
            if video_length > subvideo_length_img_prop:
                updated_frames, updated_masks = [], []
                pad_len = 10
                for f in range(0, video_length, subvideo_length_img_prop):
                    s_f = max(0, f - pad_len)
                    e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                    pad_len_s = max(0, f) - s_f
                    pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                    b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                    pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f-1], pred_flows_bi[1][:, s_f:e_f-1])
                    prop_imgs_sub, updated_local_masks_sub = self.model.img_propagation(masked_frames[:, s_f:e_f], 
                                                                        pred_flows_bi_sub, 
                                                                        masks_dilated[:, s_f:e_f], 
                                                                        'nearest')
                    updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + \
                                        prop_imgs_sub.view(b, t, 3, h, w) * masks_dilated[:, s_f:e_f]
                    updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                    
                    updated_frames.append(updated_frames_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    updated_masks.append(updated_masks_sub[:, pad_len_s:e_f-s_f-pad_len_e])
                    torch.cuda.empty_cache()

                updated_frames = torch.cat(updated_frames, dim=1)
                updated_masks = torch.cat(updated_masks, dim=1)
            else:
                b, t, _, _, _ = masks_dilated.size()
                prop_imgs, updated_local_masks = self.model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, 'nearest')
                updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
                updated_masks = updated_local_masks.view(b, t, 1, h, w)
                torch.cuda.empty_cache()            
                
        comp_frames = [None] * video_length

        neighbor_stride = neighbor_length // 2
        if video_length > subvideo_length:
            ref_num = subvideo_length // ref_stride
        else:
            ref_num = -1
        
        torch.cuda.empty_cache()
        # ---- feature propagation + transformer ----
        for f in tqdm(range(0, video_length, neighbor_stride)):
            neighbor_ids = [
                i for i in range(max(0, f - neighbor_stride),
                                    min(video_length, f + neighbor_stride + 1))
            ]
            ref_ids = get_ref_index(f, neighbor_ids, video_length, ref_stride, ref_num)
            selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
            selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
            selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
            selected_pred_flows_bi = (pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])
            
            with torch.no_grad():
                # 1.0 indicates mask
                l_t = len(neighbor_ids)
                
                # pred_img = selected_imgs # results of image propagation
                pred_img = self.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
                pred_img = pred_img.view(-1, 3, h, w)

                ## compose with input frames
                pred_img = (pred_img + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                binary_masks = masks_dilated_ori[0, neighbor_ids, :, :, :].cpu().permute(
                    0, 2, 3, 1).numpy().astype(np.uint8)  # use original mask
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] \
                        + ori_frames_inp[idx] * (1 - binary_masks[i])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else: 
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                        
                    comp_frames[idx] = comp_frames[idx].astype(np.uint8)
            
            torch.cuda.empty_cache()

        ##save composed video##

        #comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
        pil_frames =  [Image.fromarray(f) for f in comp_frames]
        # writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"),
        #                         fps, (comp_frames[0].shape[1],comp_frames[0].shape[0]))
        # for f in range(video_length):
        #     frame = comp_frames[f].astype(np.uint8)
        #     writer.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # writer.release()
       
        # torch.cuda.empty_cache()

        return pil_frames


    