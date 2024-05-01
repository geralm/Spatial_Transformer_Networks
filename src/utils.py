import torch
import numpy as np
import json

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

def compute_accuracy(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100    
def save_menu() -> bool:
    should_Save: bool = True
    print("***************************************")
    print("*    Training finished successfully   *")
    print("***************************************")
    entry = input("Do you want to save this model? [y/n]: ")
    while True:
        if entry.lower() == "y":
            should_Save = True
            break
        elif entry.lower() == "n":
            should_Save = False
            break
        else:
            entry = input("Invalid input. Please enter 'y' or 'n': ")
    return should_Save
