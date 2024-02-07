from pathlib import Path
import os, sys

ext_path = Path(__file__).parent
sys.path.insert(0, str(ext_path.resolve()))
for model in os.listdir((ext_path / "comfyui_vidmatt").resolve()):
    model_path = (ext_path / "comfyui_vidmatt" / model).resolve()
    sys.path.insert(0, str(model_path))

from comfyui_vidmatt.robust_video_matting import RobustVideoMatting
from comfyui_vidmatt.briaai_rembg import BriaaiRembg

NODE_CLASS_MAPPINGS = {
    "Robust Video Matting": RobustVideoMatting,
    "BRIAAI Matting": BriaaiRembg
}