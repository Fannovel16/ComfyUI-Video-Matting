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

download_url_template = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_{backbone}_{dtype}.torchscript"
device = get_torch_device()
class RobustVideoMatting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",), 
                "backbone": (["mobilenetv3", "resnet50"], {"default": "resnet50"}),
                "fp16": ("BOOLEAN", {"default": True}),
                "bg_color": ("STRING", {"default": "green"}),
                "batch_size": ("INT", {"min": 1, "max": 64, "default": 4})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "matting"
    CATEGORY = "Video Matting"

    def matting(self, video_frames, backbone, fp16, bg_color, batch_size):
        model_path = load_file_from_url(download_url_template.format(backbone=backbone, dtype="fp16" if fp16 else "fp32"), model_dir=CKPTS_PATH)
        model = torch.jit.load(model_path, map_location="cpu")
        model.to(device)
        video_frames, orig_num_frames, bg_color = prepare_frames_color(video_frames, bg_color, batch_size)
        bg_color = bg_color.to(device)
        if fp16:
            model.half()
            bg_color.half()
        model = torch.jit.freeze(model)
        rec, fgrs, masks = [None] * 4, [], []
        for i in range(video_frames.shape[0] // batch_size):
            input = video_frames[i*batch_size:(i+1)*batch_size].to(device)
            if fp16:
                input = input.half()
            fgr, pha, *rec = model(input, *rec, auto_downsample_ratio(*video_frames.shape[2:]))
            mask = pha.gt(0) #Remove blur
            fgr = fgr * mask + bg_color * ~mask
            fgrs.append(fgr)
            masks.append(mask.to(fgr.dtype))
        fgrs = rearrange(torch.cat(fgrs, dim=0), "n c h w -> n h w c")[:orig_num_frames].cpu().detach().float()
        masks = torch.cat(masks, dim=0)[:orig_num_frames].cpu().detach().float()
        soft_empty_cache()
        return (fgrs, masks)