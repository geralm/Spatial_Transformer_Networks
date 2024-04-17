import torch
from preprocessing import *
import preprocessing as prep
from utils import *
from model import *

def main(config:dict):  
    prep.preprocess(config)
    
    data_loader = prep.Data_Loader(config)
    train = data_loader.train_loader()
    valid = data_loader.valid_loader()
    test = data_loader.test_loader()

    print("Data loaded")
    print(f"Train: {train}")
    print(f"Test: {test}")
    run(train, test, valid,config)
if __name__ == "__main__":
    show_info()
    config= load_config() # Open the configuration file
    main(config)


