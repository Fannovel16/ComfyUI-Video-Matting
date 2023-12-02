import torch
from einops import rearrange, repeat
import os, yaml
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
from comfy.model_management import soft_empty_cache, get_torch_device

config_path = os.path.join(os.path.dirname(__file__), "./config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Video-Matting")

def auto_downsample_ratio(h, w):
    """
    Automatically find a downsample ratio so that the largest side of the resolution be 512px.
    """
    return min(512 / max(h, w), 1)
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, 'checkpoints')

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    file_name = os.path.basename(parts.path)
    if file_name is not None:
        file_name = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

download_url_template = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_{backbone}_{dtype}.torchscript"
device = get_torch_device()
class RobustVideoMatting:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_frames": ("IMAGE",), 
                "backbone": (["mobilenetv3", "resnet50"], {"default": "mobilenetv3"}),
                "fp16": ("BOOLEAN", {"default": True}),
                "batch_size": ("INT", {"min": 1, "max": 64, "default": 4})
            }
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "matting"
    CATEGORY = "Video Matting/Robust Video Matting"

    def matting(self, video_frames, backbone, fp16, batch_size):
        model_path = load_file_from_url(download_url_template.format(backbone=backbone, dtype="fp16" if fp16 else "fp32"), model_dir=config["ckpts_path"])
        model = torch.jit.load(model_path, map_location="cpu")
        model.to(device)
        video_frames = rearrange(video_frames, "n h w c -> n c h w")
        if fp16:
            model.half()
        model = torch.jit.freeze(model)
        orig_num_frames = video_frames.shape[0]
        pad_frames = repeat(video_frames[-1:], "1 c h w -> n c h w", n=batch_size - (orig_num_frames % batch_size))
        video_frames = torch.cat([video_frames, pad_frames], dim=0)
        rec, fgrs, masks = [None] * 4, [], []
        for i in range(video_frames.shape[0] // batch_size):
            input = video_frames[i*batch_size:(i+1)*batch_size].to(device)
            if fp16:
                input = input.half()
            fgr, pha, *rec = model(input, *rec, auto_downsample_ratio(*video_frames.shape[2:]))
            fgr = fgr * pha.gt(0)
            fgrs.append(fgr.float())
            masks.append(pha.float())
        fgrs = rearrange(torch.cat(fgrs, dim=0), "n c h w -> n h w c")[:orig_num_frames].cpu().detach()
        masks = torch.cat(masks, dim=0)[:orig_num_frames].cpu().detach()
        soft_empty_cache()
        return (fgrs, masks)

NODE_CLASS_MAPPINGS = {
    "Robust Video Matting": RobustVideoMatting
}