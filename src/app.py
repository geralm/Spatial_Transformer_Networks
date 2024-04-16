import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from preprocessing import *
import preprocessing as prep
from utils import *
from model import *
import pandas as pd

def main(config:dict):  
    prep.preprocess(config)
    
    data_loader = prep.Data_Loader(config)
    train_loader = data_loader.train_loader()
    valid_loader = data_loader.valid_loader()
    test_loader = data_loader.test_loader()

   
    
    print("Data loaded")
    print(f"Train: {train_loader}")
    print(f"Test: {valid_loader}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #run(device, train, test)
if __name__ == "__main__":
    show_info()
    config= load_config() # Open the configuration file
    main(config)


