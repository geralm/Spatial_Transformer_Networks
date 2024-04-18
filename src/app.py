import torch
from preprocessing import *
import preprocessing as prep
from utils import *
from model import *
from Tester import * 
from Trainer import *

def main(config:dict):  
    saveModelPath:str = config["model"]["SAVE_MODEL_PATH"]
    autoSave:bool = config["model"]["AUTO_SAVE"]
    show_info()
    # Preprocess the data 
    prep.preprocess(config)
    # Load the data
    print("Loading the data...")
    data_loader = prep.Data_Loader(config)
    train = data_loader.train_loader()
    valid = data_loader.valid_loader()
    test = data_loader.test_loader()
    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    #Trainer object
    trainer  = Trainer(
        train_loader=train,
        validation_loader=valid,
        model=model,
        device=device,
        settings=config
        )
    tester = Tester(
            test_loader=test,
            model=model,
            device=device
        )

    is_training= config["model"]["TRAIN"]
    if is_training: # If the model is going to be trained, otherwise you need to have pretrained model
        print("Training the model...")
        model = trainer.run()
        if autoSave:
            torch.save(model.state_dict(), saveModelPath)
            print("Model saved!")
        else:
            if save_menu():
                torch.save(model.state_dict(), saveModelPath)
                print("Model saved!")
    else:
        # Load the model if it is not going to be trained
        # but you should have pretrained one
        try:
            model.load_state_dict(torch.load(saveModelPath))
            print("Model loaded sucessfully")
        except:
            print(f"You need a pretreined model: Model not found in the directory {saveModelPath} or the model class has changed")
            return -1        
    is_testing = config["model"]["TEST"]
    if is_testing:
        tester.run()
        tester.random_test()
        tester.visualize_stn()
if __name__ == "__main__":
    config=load_config() # Open the configuration file
    main(config)


