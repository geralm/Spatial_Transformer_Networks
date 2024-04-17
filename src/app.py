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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config["model"]["TRAIN"]:
        model, device = run(train, test, valid,config)
        if config["model"]["SAVE"]:
            save_model(model, config["model"]["SAVE_MODEL_PATH"])
    if config["model"]["LOAD_MODEL"]:
        model = load_model(config, device)
    random_test(model, test, device)
    visualize_stn(test, model, device)
    plt.ioff()
    plt.show()
if __name__ == "__main__":
    show_info()
    config= load_config() # Open the configuration file
    main(config)


