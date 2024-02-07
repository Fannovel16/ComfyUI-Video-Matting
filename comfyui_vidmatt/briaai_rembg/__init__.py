import torch, os
from PIL import Image
from comfyui_vidmatt.briaai_rembg.arch import BriaRMBG
import torch.nn.functional as F
from torchvision.transforms.functional import normalize

import torch
from einops import rearrange, repeat
from comfy.model_management import soft_empty_cache, get_torch_device
from PIL import ImageColor

from comfyui_vidmatt.utils import CKPTS_PATH, load_file_from_url, prepare_frames_color

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)

download_url = "https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth"
device = get_torch_device()
model_input_size = [1024,1024]

class BriaaiRembg:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "version": (["v1.4"], {"default": "v1.4"}),
                "fp16": ("BOOLEAN", {"default": True}),
                "bg_color": ("STRING", {"default": "green"}),
                "batch_size": ("INT", {"min": 1, "max": 64, "default": 4})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "matting"
    CATEGORY = "Video Matting"


    def matting(self, video_frames, version, fp16, bg_color, batch_size):
        model_path = load_file_from_url(download_url, file_name=f"briaai_rmbg_{version}.pth", model_dir=CKPTS_PATH)
        model = BriaRMBG()
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.to(device).eval()

        video_frames, orig_num_frames, bg_color = prepare_frames_color(video_frames, bg_color, batch_size)
        bg_color = bg_color.to(device)
        orig_frame_size = video_frame.shape[2:4]
        video_frames = F.interpolate(video_frames, size=model_input_size, mode='bilinear')
        if fp16:
            model.half()
            bg_color.half()
        
        fgrs, masks = [], []
        for i in range(video_frames.shape[0] // batch_size):
            input = video_frames[i*batch_size:(i+1)*batch_size].to(device)
            if fp16:
                input = input.half()
            mask = model(normalize(input,[0.5,0.5,0.5],[1.0,1.0,1.0]))[0][0]
            mask = (mask-mask.min())/(mask.max()-mask.min()) #This is sharp enough
            fgr = input * mask + bg_color * (1 - mask) 
            fgrs.append(fgr)
            masks.append(mask.to(fgr.dtype))
        fgrs = F.interpolate(torch.cat(fgrs, dim=0), size=orig_frame_size).float().cpu().detach()
        fgrs = rearrange(fgrs, "n c h w -> n h w c")[:orig_num_frames]
        masks = F.interpolate(torch.cat(masks, dim=0), size=orig_frame_size).float().cpu().detach()
        masks = masks[:orig_num_frames]
        soft_empty_cache()
        return (fgrs, masks)
