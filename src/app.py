import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from dataloader import Data_Loader
from utils import *
from model import *

def main(config:dict):     
    #extract_data()
    #move_to_preprocessing()
    train_path = config["data_loader"]["train"]["path"]
    train_label = config["data_loader"]["train"]["label"]
    train_batch = config["data_loader"]["train"]["batch_size"]

    test_path = config["data_loader"]["test"]["path"]
    test_batch = config["data_loader"]["test"]["batch_size"]
    test_label = config["data_loader"]["test"]["label"]
    img_size = config["data_loader"]["img_size"]
    numworkers = 4

    train = Data_Loader(train_path, train_label, img_size,
                             train_batch, True, numworkers).loader()
    test = Data_Loader(test_path, test_label, img_size,
                    test_batch, numworkers, False).loader()
    print("Data loaded")
    print(f"Train: {train}")
    print(f"Test: {test}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(device, train, test)
if __name__ == "__main__":
    show_info()
    config= load_config() # Open the configuration file
    main(config)


