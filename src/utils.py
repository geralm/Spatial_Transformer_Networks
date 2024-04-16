import torch
import numpy as np
import json
import zipfile
import shutil
import os
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
def data_preprocessing():
    pass

def move_to_preprocessing():
    """
    Move the mask folder, the image folder, and CelebA-HQ-to-CelebA-mapping.txt (remove 1st line in advance) under ./Data_preprocessing
    """
    config = load_config()
    paths = config["data_loader"]
    imgfolder = "CelebA-HQ-img"    
    maskfolder = "CelebAMask-HQ-mask-anno"
    mappingtxt= "CelebA-HQ-to-CelebA-mapping.txt"   
    path = Path(paths["path"])
    dir_name = Path(paths["dir_name"])

    original_image_path = path / dir_name / imgfolder
    original_mask_path = path / dir_name / maskfolder
    original_mapping_path = path / dir_name / mappingtxt
    target_path = "./src/Data_preprocessing"
    if Path(os.path.join(target_path, imgfolder)).is_dir():
        print("The preprocessing data already exist")
    else:
        print("Moving data to preprocess...")
        shutil.move(original_image_path, target_path)
        shutil.move(original_mask_path, target_path)
        shutil.move(original_mapping_path, target_path)
        with open(os.path.join(target_path, mappingtxt), 'r') as fin:
            data = fin.read().splitlines(True)
        with open(os.path.join(target_path, mappingtxt), 'w') as fout:
            fout.writelines(data[1:])

def extract_data():
    """
    Call this function to extract CelebAMask HQ data if the directory doesn't exist
    creates ones.
    Params: No params
    Return: 0 if everything okay
    """
    data_loader_info = load_config()["data_loader"]
    path = Path(data_loader_info["path"])
    zipname = data_loader_info["zipname"]
    dirname = path / data_loader_info["dir_name"]
    if dirname.is_dir() :
        print(f"{dirname} directory exist")
    else: 
        print(f"Did not find {dirname} directory, creating one...")
        dirname.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(path / zipname, "r") as zip_ref:
            print("Unzipping CelebAMask...") 
            zip_ref.extractall(dirname)
    return 0