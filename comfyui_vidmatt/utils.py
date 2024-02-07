import os, yaml
from torch.hub import download_url_to_file, get_dir
from urllib.parse import urlparse
from einops import rearrange, repeat
import torch
from PIL import ImageColor

config_path = os.path.join(os.path.dirname(__file__), "../config.yaml")
if os.path.exists(config_path):
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
else:
    raise Exception("config.yaml file is neccessary, plz recreate the config file by downloading it from https://github.com/Fannovel16/ComfyUI-Video-Matting")
CKPTS_PATH = os.path.join(os.path.join(os.path.dirname(__file__), '..'), config["ckpts_path"])

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
    if file_name is None:
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file

def prepare_frames_color(video_frames, bg_color, batch_size):
    orig_num_frames = video_frames.shape[0]
    video_frames = rearrange(video_frames, "n h w c -> n c h w")
    pad_frames = repeat(video_frames[-1:], "1 c h w -> n c h w", n=batch_size - (orig_num_frames % batch_size))
    video_frames = torch.cat([video_frames, pad_frames], dim=0)
    
    bg_color = torch.Tensor(ImageColor.getrgb(bg_color)[:3]).float() / 255.
    bg_color = repeat(bg_color, "c -> n c 1 1", n=batch_size)
    
    return video_frames, orig_num_frames, bg_color
