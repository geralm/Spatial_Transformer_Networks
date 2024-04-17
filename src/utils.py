import torch
import numpy as np
import json
from pathlib import Path
def show_info():
    print("Versión de NumPy:", np.__version__)
    print("Versión de PyTorch:", torch.__version__)
    print(f'cuda enable: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'cuda version: {torch.version.cuda}')
        print(f'current_device: {torch.cuda.current_device()}')
        print(f'device: {torch.cuda.device(0)}')
        print(f'device_count: {torch.cuda.device_count()}')
        print(f'get_device_name: {torch.cuda.get_device_name(0)}')
    else:
        print('cuda not available')
def load_config()->dict:
    """
    Open the configuration file config.json
    Returns the data in a dict format
    """
    f = open('./src/config.json')
    # returns JSON object as 
    # a dictionary
    data = json.load(f)
    return data
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp
