
import pandas as pd
import zipfile
import os
import numpy as np
import torch.utils.data as utils_data
from PIL import Image
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
def extract_data(zip_path):
    """
    Call this function to extract Celeba
    Params: 
    """
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        print("Unzipping Celeba...") 
        zip_ref.extractall("./celeba")
def read_attributes(path:str):
    """
    Read the attributes from the file attr.txt
    """
    df = pd.read_csv(path, sep="\s+", skiprows=1, usecols=['Male'])
    # Make 0 (female) & 1 (male) labels instead of -1 & 1
    df.loc[df['Male'] == -1, 'Male'] = 0
    return df
def read_partition(path:str): 
    """
    Read the partition from the file eval_partition.txt
    """
    df = pd.read_csv(path, sep="\s+", skiprows=0, header=None)
    df.columns = ['Filename', 'Partition']
    df = df.set_index('Filename')
    return df
def split(to_csv_path:str, 
          train_csv, 
          valid_csv, 
          test_csv ,  
          df1, df2):
    """
    Split the data into train, validation and test
    """
    df3 = df1.merge(df2, left_index=True, right_index=True)
    df3.to_csv(to_csv_path)
    df4 = pd.read_csv(to_csv_path, index_col=0)
    df4.loc[df4['Partition'] == 0].to_csv(train_csv)
    df4.loc[df4['Partition'] == 1].to_csv(valid_csv)
    df4.loc[df4['Partition'] == 2].to_csv(test_csv)
    return df4

class CelebaDataset(utils_data.Dataset):
    """Custom Dataset for loading CelebA face images"""
    def __init__(self, csv_path, img_dir, transform=None):
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        if self.transform is not None:
            img = self.transform(img)
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
def show_image(path:str)->None:
        """
        Show an image
        """
        try:
            img = Image.open(path)
            print(np.asarray(img, dtype=np.uint8).shape)
            plt.imshow(img)
            plt.show()
        except:
            print("Could not open the image")
            return -1 
        return 0  
class Data_Loader(utils_data.DataLoader):
    def __init__(self, settings:dict):
        self.train_file = settings["data"]["TRAIN_FILE"] 
        self.val_file = settings["data"]["VAL_FILE"]
        self.test_file = settings["data"]["TEST_FILE"]
        self.img_dir = settings["data"]["IMG_DIR"]
        self.batch = settings["data"]["BATCH_SIZE"]
        self.num_workers = settings["data"]["NUM_WORKERS"]
        self.custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                       transforms.Resize((128, 128)),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.4914, 0.4822, 0.4465), (0.5, 0.5, 0.5))])
        # Note that transforms.ToTensor()
        # already divides pixels by 255. internally

    def _train_dataset(self):
        return CelebaDataset(csv_path=self.train_file,
                              img_dir=self.img_dir,
                              transform=self.custom_transform)
    def _valid_dataset(self):
        return CelebaDataset(csv_path= self.val_file,
                              img_dir=self.img_dir,
                              transform=self.custom_transform)
    def _test_dataset(self):
        return CelebaDataset(csv_path= self.test_file,
                             img_dir=self.img_dir,
                             transform=self.custom_transform)
    def train_loader(self):
        return utils_data.DataLoader(dataset=self._train_dataset(),
                                     batch_size=self.batch,
                                     shuffle=True,
                                     num_workers=self.num_workers)
    def valid_loader(self):
        return utils_data.DataLoader(dataset=self._valid_dataset(),
                                     batch_size=self.batch,
                                     shuffle=False,
                                     num_workers=self.num_workers)
    def test_loader(self):
        return utils_data.DataLoader(dataset=self._test_dataset(),
                                     batch_size=self.batch,
                                     shuffle=False,
                                     num_workers=self.num_workers)


def preprocess(config):
    """
    This function runs the preprocessing steps
    1) Extract the data
    2) Read the attributes
    3) Read the partition
    4) Split the data
    5) Save the dataframes to a CSV file
    """
    zippath = config['data']['ZIPNAME']   
    img_dir = config['data']['IMG_DIR']
    attr_file = config['data']['ATTR_FILE']
    partition_file = config['data']['PARTITION_FILE']
    # CSV names
    to_csv = config['data']['TO_CSV']
    train_path = config['data']['TRAIN_FILE']
    test_path = config['data']['TEST_FILE']
    val_path = config['data']['VAL_FILE']
    if os.path.isdir(img_dir) :
        print(f"{img_dir} directory exist")
    else: 
        print(f"Did not find {img_dir} directory, creating one...")
        extract_data(zippath)

    if(os.path.exists(to_csv)):
        print(f"{to_csv} exist")
    else:
        print(f"{to_csv} does not exist, creating one...")
        df1 = read_attributes(attr_file)
        df2 = read_partition(partition_file)
        df3 = split(to_csv, train_path, val_path, test_path, df1, df2)
        print(df3.head()) 
        print(f"Dataframes saved to {to_csv}")                                            