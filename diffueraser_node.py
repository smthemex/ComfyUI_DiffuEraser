# ComfyUI-native DiffuEraser nodes.
# Inference runs entirely on the comfy core stack (ModelPatcher / model_management
# / comfy attention / comfy VAE); no diffusers or transformers imports here.
import copy
import gc
import os

import numpy as np
import torch

from comfy_api.latest import  io
import folder_paths

from .node_utils import load_images, tensor2pil_list, image2masks
from .run_diffueraser import load_diffueraser, load_propainter

MAX_SEED = np.iinfo(np.int32).max
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# add checkpoints dir
DiffuEraser_weigths_path = os.path.join(folder_paths.models_dir, "DiffuEraser")
if not os.path.exists(DiffuEraser_weigths_path):
    os.makedirs(DiffuEraser_weigths_path)
folder_paths.add_model_folder_path("DiffuEraser", DiffuEraser_weigths_path)


class Propainter_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Propainter_Loader",
            display_name="Propainter_Loader",
            category="DiffuEraser",
            inputs=[
                io.Combo.Input("propainter", options=["none"] + folder_paths.get_filename_list("DiffuEraser")),
                io.Combo.Input("flow", options=["none"] + folder_paths.get_filename_list("DiffuEraser")),
                io.Combo.Input("fix_raft", options=["none"] + folder_paths.get_filename_list("DiffuEraser")),
                io.Combo.Input("device", options=["cpu", "cuda", "mps"]),
            ],
            outputs=[
                io.Custom("Propainter_Loader").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, propainter, flow, fix_raft, device) -> io.NodeOutput:
        ProPainter_path = folder_paths.get_full_path("DiffuEraser", propainter) if propainter != "none" else None
        flow_path = folder_paths.get_full_path("DiffuEraser", flow) if flow != "none" else None
        fix_raft_path = folder_paths.get_full_path("DiffuEraser", fix_raft) if fix_raft != "none" else None
        if fix_raft_path is None or flow_path is None or ProPainter_path is None:
            raise ValueError("need load all models")
        model = load_propainter(fix_raft_path, flow_path, ProPainter_path, device=device)
        return io.NodeOutput(model)


class DiffuEraser_Loader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffuEraser_Loader",
            display_name="DiffuEraser_Loader",
            category="DiffuEraser",
            inputs=[
                io.Combo.Input("vae", options=folder_paths.get_filename_list("vae")),
                io.Combo.Input("lora", options=["none"] + folder_paths.get_filename_list("loras")),
                io.Combo.Input("dtype", options=["fp16", "fp32"], default="fp16"),
            ],
            outputs=[
                io.Custom("DiffuEraser_Loader").Output(display_name="model"),
            ],
        )

    @classmethod
    def execute(cls, vae, lora, dtype) -> io.NodeOutput:
        vae_path = folder_paths.get_full_path("vae", vae)
        pcm_lora_path = folder_paths.get_full_path("loras", lora) if lora != "none" else None
        torch_dtype = torch.float16 if dtype == "fp16" else torch.float32
        model = load_diffueraser(vae_path, pcm_lora_path, device=device, dtype=torch_dtype)
        gc.collect()
        torch.cuda.empty_cache()
        return io.NodeOutput(model)


class DiffuEraser_PreData(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffuEraser_PreData",
            display_name="DiffuEraser_PreData",
            category="DiffuEraser",
            inputs=[
                io.Image.Input("images"),
                io.String.Input("seg_repo", default="briaai/RMBG-2.0"),
                io.Image.Input("video_mask_image", optional=True),
                io.Mask.Input("video_mask", optional=True),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
            ],
        )

    @classmethod
    def execute(cls, images, seg_repo, video_mask_image=None, video_mask=None) -> io.NodeOutput:
        _, height, width, _ = images.size()
        height, width = (height - height % 8, width - width % 8)
        video_image = tensor2pil_list(images, width, height)

        if video_mask is None and video_mask_image is None and seg_repo:    # use rmbg or BiRefNet to make video to masks
            print("*********** Use input video and repo to make masks **************")
            init_mask = image2masks(seg_repo, video_image)
        elif video_mask_image is not None:
            if not isinstance(video_mask_image, torch.Tensor):
                raise ValueError("video_mask_image is not a normal comfyUI image tensor, need a shape like  b,h,w,c")
            else:
                init_mask = tensor2pil_list(video_mask_image, width, height)

        elif video_mask is not None:
            if isinstance(video_mask, torch.Tensor) and len(video_mask) > 3:
                raise ValueError("video_mask is not a normal comfyUI mask tensor, need a shape like  b,h,w")
            init_mask = tensor2pil_list(
                video_mask.reshape((-1, 1, video_mask.shape[-2], video_mask.shape[-1])).movedim(1, -1).expand(-1, -1, -1, 3),
                width, height)
        else:
            raise ValueError("no video_mask,you can enable video2mask and fill a rmbg or BiRefNet repo to generate mask from video_image,or link video_mask from other node")

        if len(init_mask) != len(video_image):
            if len(init_mask) == 1:
                init_mask = init_mask * len(video_image)  # if use one mask to inpaint all frames
            else:
                if len(init_mask) > len(video_image):
                    init_mask = init_mask[:len(video_image)]
                    print("init_mask length:", len(init_mask), "video_image length:", len(video_image))
                else:
                    init_mask = init_mask + init_mask[:len(video_image) - len(init_mask)]
                    print("init_mask length:", len(init_mask), "video_image length:", len(video_image))
        cond = {"init_mask": init_mask, "video_image": video_image, "height": height, "width": width}
        return io.NodeOutput(cond)


class Propainter_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="Propainter_Sampler",
            display_name="Propainter_Sampler",
            category="DiffuEraser",
            inputs=[
                io.Custom("Propainter_Loader").Input("model"),
                io.Conditioning.Input("conditioning"),
                io.Float.Input("fps", force_input=True),
                io.Int.Input("video_length", default=10, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("mask_dilation_iter", default=2, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("ref_stride", default=10, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("neighbor_length", default=10, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("subvideo_length", default=50, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Image.Output(display_name="images"),
            ],
        )

    @classmethod
    def execute(cls, model, conditioning, fps, video_length, mask_dilation_iter, ref_stride, neighbor_length, subvideo_length) -> io.NodeOutput:
        model.to(device)
        conditioning["fps"] = fps
        conditioning["video_length"] = video_length
        conditioning["mask_dilation_iter"] = mask_dilation_iter

        Propainter_img = model.forward(copy.deepcopy(conditioning["video_image"]), copy.deepcopy(conditioning["init_mask"]),
                                       load_videobypath=False, video_length=video_length, height=conditioning["height"],
                                       width=conditioning["width"], ref_stride=ref_stride, neighbor_length=neighbor_length,
                                       subvideo_length=subvideo_length, mask_dilation=mask_dilation_iter, save_fps=fps)
        conditioning["prioris"] = Propainter_img
        model.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        images = load_images(Propainter_img)
        return io.NodeOutput(conditioning, images)


class DiffuEraser_Sampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="DiffuEraser_Sampler",
            display_name="DiffuEraser_Sampler",
            category="DiffuEraser",
            inputs=[
                io.Custom("DiffuEraser_Loader").Input("model"),
                io.Conditioning.Input("positive"),
                io.Conditioning.Input("conditioning"),
                io.Conditioning.Input("negative", optional=True),
                io.Int.Input("steps", default=2, min=1, max=1024, step=1, display_mode=io.NumberDisplay.number),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED, display_mode=io.NumberDisplay.number),
                io.Int.Input("nframes", default=22, min=4, max=64, step=1, display_mode=io.NumberDisplay.number),
                io.Float.Input("guidance_scale", default=0.0, min=0.0, max=30.0, step=0.1),
                io.Boolean.Input("save_result_video", default=False),
                io.Boolean.Input("blended", default=False),
            ],
            outputs=[
                io.Image.Output(display_name="image"),
            ],
        )

    @classmethod
    def execute(cls, model, positive, conditioning, steps, seed, nframes, guidance_scale,
                save_result_video, blended, negative=None) -> io.NodeOutput:
        missing = [k for k in ("video_image", "init_mask", "prioris") if k not in conditioning]
        if missing:
            raise ValueError(
                f"conditioning is missing {missing}: the workflow must chain "
                "DiffuEraser_PreData -> Propainter_Loader -> Propainter_Sampler -> DiffuEraser_Sampler. "
                "Propainter_Sampler writes 'prioris' (and fps / mask_dilation_iter) into the conditioning; "
                "connecting PreData straight to DiffuEraser_Sampler is not supported."
                "（工作流必须经过 Propainter_Sampler 节点，它会向 conditioning 写入 prioris）"
            )
        positive_embeds = positive[0][0]
        negative_embeds = negative[0][0] if negative is not None else None
        if guidance_scale > 1.0 and negative_embeds is None:
            raise ValueError("guidance_scale > 1 requires the negative conditioning input")

        frames = model.run_inference(
            conditioning["video_image"], conditioning["init_mask"], conditioning["prioris"],
            positive_embeds, negative_embeds=negative_embeds,
            num_frames=nframes, num_inference_steps=steps, guidance_scale=guidance_scale,
            seed=seed, mask_dilation_iter=conditioning.get("mask_dilation_iter", 4),
            blended=blended, if_save_video=save_result_video,
            output_path=folder_paths.get_output_directory(), fps=conditioning.get("fps", 24))

        # two-stage workflow: free the diffusion models after sampling
        model.unload()
        gc.collect()
        torch.cuda.empty_cache()
        images = load_images(frames)
        return io.NodeOutput(images)


