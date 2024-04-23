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
        self.fc2 = nn.Linear(3*128*128, 2)
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_dim, 8, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
        )
        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 28 * 28, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_() 
        self.fc_loc[2].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0] , dtype=torch.float)
            )
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1,  10 * 28 * 28)  
        # xs = F.normalize(xs, dim=-1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3) # Theta size [N x 2 x 3] 
        
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        #,mode="bilinear", padding_mode="border",
        x = F.grid_sample(x, grid, align_corners=False)
        return x
    def forward(self, x):
        x = self.stn(x)
        x = x.view(-1, 3*128*128)
        logits  = self.fc2(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
def build_model(config:dict):
    channels = config["model"]["CHANNELS"]
    return STN(channels)