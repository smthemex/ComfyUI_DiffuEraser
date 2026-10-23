# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import torch
from PIL import Image
import numpy as np
import cv2
import time
from comfy.utils import common_upscale, ProgressBar
import torchvision.transforms as transforms
import gc

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def image2masks(repo, video_image):
    start_time = time.time()
    from transformers import AutoModelForImageSegmentation
    model = AutoModelForImageSegmentation.from_pretrained(repo, trust_remote_code=True)
    torch.set_float32_matmul_precision(['high', 'highest'][0])
    model.to(device)
    model.eval()
    # Data settings
    image_size = (1024, 1024)
    transform_image = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    masks = []
    for img in video_image:
        input_images = transform_image(img).unsqueeze(0).to(device)
        # Prediction
        with torch.no_grad():
            preds = model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(img.size)
        masks.append(mask.convert('RGB'))
    end_time = time.time()
    load_time = end_time - start_time
    print(f"image2masks infer time: {load_time:.4f} s")
    model.to('cpu')
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def tensor_to_pil(tensor):
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image


def tensor2pil_list(image, width, height):
    B, _, _, _ = image.size()
    if B == 1:
        ref_image_list = [tensor2pil_upscale(image, width, height)]
    else:
        img_list = list(torch.chunk(image, chunks=B))
        ref_image_list = [tensor2pil_upscale(img, width, height) for img in img_list]
    return ref_image_list


def tensor2pil_upscale(img_tensor, width, height):
    samples = img_tensor.movedim(-1, 1)
    img = common_upscale(samples, width, height, "bilinear", "center")
    samples = img.movedim(1, -1)
    img_pil = tensor_to_pil(samples)
    return img_pil


def images_generator(img_list: list,):
    # get img size
    sizes = {}
    for image_ in img_list:
        if isinstance(image_, Image.Image):
            count = sizes.get(image_.size, 0)
            sizes[image_.size] = count + 1
        elif isinstance(image_, np.ndarray):
            count = sizes.get(image_.shape[:2][::-1], 0)
            sizes[image_.shape[:2][::-1]] = count + 1
        else:
            raise ValueError("unsupported image list, must be pil or cv2")
    size = max(sizes.items(), key=lambda x: x[1])[0]
    yield size[0], size[1]

    # any to tensor
    def load_image(img_in):
        if isinstance(img_in, Image.Image):
            img_in = img_in.convert("RGB")
            i = np.array(img_in, dtype=np.float32)
            i = torch.from_numpy(i).div_(255)
            if i.shape[0] != size[1] or i.shape[1] != size[0]:
                i = torch.from_numpy(i).movedim(-1, 0).unsqueeze(0)
                i = common_upscale(i, size[0], size[1], "lanczos", "center")
                i = i.squeeze(0).movedim(0, -1).numpy()
            return i
        elif isinstance(img_in, np.ndarray):
            i = cv2.cvtColor(img_in, cv2.COLOR_BGR2RGB).astype(np.float32)
            i = torch.from_numpy(i).div_(255)
            return i
        else:
            raise ValueError("unsupported image list, must be pil, cv2 or tensor")

    total_images = len(img_list)
    processed_images = 0
    pbar = ProgressBar(total_images)
    images = map(load_image, img_list)
    try:
        prev_image = next(images)
        while True:
            next_image = next(images)
            yield prev_image
            processed_images += 1
            pbar.update_absolute(processed_images, total_images)
            prev_image = next_image
    except StopIteration:
        pass
    if prev_image is not None:
        yield prev_image


def load_images(img_list: list,):
    gen = images_generator(img_list)
    (width, height) = next(gen)
    images = torch.from_numpy(np.fromiter(gen, np.dtype((np.float32, (height, width, 3)))))
    if len(images) == 0:
        raise FileNotFoundError(f"No images could be loaded .")
    return images
