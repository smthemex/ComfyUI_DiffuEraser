import torch
import os
import folder_paths

# Thin launcher: ProPainter stays self-contained; DiffuEraser uses the comfy-native core.
from .propainter.inference import Propainter, get_device


def load_propainter(fix_raft_path, flow_path, ProPainter_path, device="cpu"):
    model = Propainter(device)
    model.load_propainter(fix_raft_path, flow_path, ProPainter_path)
    return model


def load_diffueraser(vae_path, pcm_lora_path, device=None, dtype=torch.float16):
    from .diffueraser_core import load_diffueraser_core
    diffueraser_dir = os.path.join(folder_paths.models_dir, "DiffuEraser")
    repo_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sd15_repo")
    scheduler_config = os.path.join(repo_dir, "scheduler", "scheduler_config.json")
    return load_diffueraser_core(diffueraser_dir, vae_path, pcm_lora_path,
                                 repo_dir, scheduler_config, device=device, dtype=dtype)
