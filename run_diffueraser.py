import torch
import os 
import time
import random
from .libs.diffueraser import DiffuEraser
from .propainter.inference import Propainter, get_device
import folder_paths
import gc


def load_diffueraser(sd_repo,pre_model_path, ckpt_path,original_config_file,pcm_lora_path,device):
    device = get_device()
    model=DiffuEraser(device)
    model.load_model(sd_repo, pre_model_path,ckpt_path,original_config_file, pcm_lora_path,)
    return model


def load_propainter(fix_raft_path,flow_path,ProPainter_path,device="cpu"):
    model=Propainter(device)   
    model.load_propainter(fix_raft_path,flow_path,ProPainter_path)
    return model





