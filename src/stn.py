#########################################
### STN (Spatial Transformer Network) ###
#########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
class STN(nn.Module): 
    def __init__(self, channels):
        super(STN, self).__init__()
        in_dim = channels
         # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(128*16*16, 256), 
            nn.ReLU(True),
            nn.Linear(256, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0] , dtype=torch.float))
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        batch_size = x.size(0)
        xs = xs.view(-1, 128*16*16)  
        theta = self.fc_loc(xs)

        # print(f"Theta before  view {theta.size()}")
        theta = theta.view(-1, 2, 3)
        # print(f"Theta after  view {theta.size()} vs {x.size()}")
        grid = F.affine_grid(theta, x.size(),align_corners=True)
        x = F.grid_sample(x, grid,align_corners=True)

        return x
    def forward(self, x):
        return self.stn(x)
def build_model(config:dict):
    channels = config["model"]["CHANNELS"]
    return STN(channels)