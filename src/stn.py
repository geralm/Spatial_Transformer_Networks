#########################################
### STN (Spatial Transformer Network) ###
#########################################
import torch
import torch.nn as nn
import torch.nn.functional as F
class STN(nn.Module): 
    def __init__(self, channels, img_size):
        super(STN, self).__init__()
        in_dim = channels
        self._imsize = img_size
        self.fc2 = nn.Linear(3*self._imsize*self._imsize, 2)
        
        self.localization = nn.Sequential(nn.Conv2d(in_dim, 64, kernel_size=7),
                                          nn.BatchNorm2d(64),
                                          nn.ReLU(True),
                                          nn.Conv2d(64, 64*4, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                          nn.Conv2d(64*4, 128*4, kernel_size=5),
                                          nn.MaxPool2d(2, stride=2),
                                          nn.ReLU(True),
                                            )

        self.fc_loc = nn.Sequential(
                                    nn.Linear( 512*11*11, 512),
                                    nn.ReLU(True),
                                    nn.Linear(512 , 2*3)
                                    )
                       
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_() 
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0] , dtype=torch.float)
            )
    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1,  512*11*11)  
        # xs = F.normalize(xs, dim=-1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3) # Theta size [N x 2 x 3] 
        
        grid = F.affine_grid(theta, x.size())
        #,mode="bilinear", padding_mode="border",
        x = F.grid_sample(x, grid)
        return x
    def forward(self, x):
        x = self.stn(x)
        x = x.view(-1, 3*self._imsize*self._imsize)
        logits  = self.fc2(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas
def build_model(config:dict):
    channels = config["model"]["CHANNELS"]
    img_size = config["data"]["IMG_SIZE"]
    return STN(channels, img_size=img_size)