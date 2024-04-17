##########################
### MODEL
##########################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from utils import convert_image_np
import time
plt.ion() # Interactive mode

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Net(nn.Module):
    def __init__(self, block, layers, num_classes, channels):
        self.inplanes = 64
        in_dim = channels
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1, padding=2)
        self.fc = nn.Linear(2048 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2. / n)**.5)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(in_dim, 64, kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(64, 16, kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(16*28*28, 32), 
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0] , dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        xs = self.localization(x)
        batch_size = x.size(0)
        xs = xs.view(batch_size, -1)  
        theta = self.fc_loc(xs)

        # print(f"Theta before  view {theta.size()}")
        theta = theta.view(-1, 2, 3)
        # print(f"Theta after  view {theta.size()} vs {x.size()}")
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid,align_corners=False)

        return x


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    

    def forward(self, x):
        x = self.stn(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        logits = self.fc(x)
        probas = F.softmax(logits, dim=1)
        return logits, probas

def build(config:dict):
    """Constructs a Net-101 model."""
    NUM_CLASSES: int = config["model"]["NUM_CLASSES"]
    LAYERS: list = config["model"]["LAYERS"]
    CHANNELS: int = config["model"]["CHANNELS"]
    model = Net(block=Bottleneck, 
                   layers=LAYERS,
                   num_classes=NUM_CLASSES,
                   channels=CHANNELS)
    return model
def train(device, train_loader, epoch, model, optimizer, total_epochs):
    model.train()
    cost_fn = torch.nn.CrossEntropyLoss() 
    for batch_idx, (features, targets) in enumerate(train_loader):
        features, targets = features.to(device), targets.to(device)       
            ### FORWARD AND BACK PROP
        logits, probas = model(features)
        cost = cost_fn(logits, targets)
        optimizer.zero_grad()     
        cost.backward()
            ### UPDATE MODEL PARAMETERS
        optimizer.step()
        ### LOGGING
        if not batch_idx % 50:
            print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                %(epoch+1,total_epochs , batch_idx, 
                    len(train_loader), cost))
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

def test(device, model,train_loader, valid_loader, epoch, total_epochs):
    with torch.no_grad(): # save memory during inference
        model.eval()        
        print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
            epoch+1, total_epochs, 
            compute_accuracy(model, train_loader, device=device),
            compute_accuracy(model, valid_loader, device=device)))
# We want to visualize the output of the spatial transformers layer
# after the training, we visualize a batch of input images and
# the corresponding transformed batch using STN.


def visualize_stn(test_loader, model, device):
    with torch.no_grad():
        # Get a batch of training data
        data = next(iter(test_loader))[0].to(device)

        input_tensor = data.cpu()
        transformed_input_tensor = model.stn(data).cpu()

        in_grid = convert_image_np(
            torchvision.utils.make_grid(input_tensor))

        out_grid = convert_image_np(
            torchvision.utils.make_grid(transformed_input_tensor))

        # Plot the results side-by-side
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(in_grid)
        axarr[0].set_title('Dataset Images')

        axarr[1].imshow(out_grid)
        axarr[1].set_title('Transformed Images')
def random_test(model, test_loader, device):
    for batch_idx, (features, targets) in enumerate(test_loader):
        features = features
        targets = targets
        break
    model.eval()
    logits, probas = model(features.to(device)[0, None])
    print('Probability Female %.2f%%' % (probas[0][0]*100))
    
def run(train_loader, test_loader, val_loader, config):
    SEED = config["model"]["RANDOM_SEED"]
    LEARNING_RATE = config["model"]["LEARNING_RATE"]
    NUM_EPOCHS = config["model"]["NUM_EPOCHS"]
    torch.manual_seed(SEED)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Building model and moving to {DEVICE}")
    model = build(config)
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)  
    start_time = time.time()
    for epoch in range(NUM_EPOCHS):
        train(DEVICE, train_loader, epoch, model, optimizer, NUM_EPOCHS)
        test(DEVICE, model, train_loader, val_loader, epoch, NUM_EPOCHS)
        print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))    
    with torch.no_grad(): # save memory during inference
        print('Test accuracy: %.2f%%' % (compute_accuracy(model, test_loader, device=DEVICE)))
    random_test(model, test_loader, DEVICE)
    visualize_stn(test_loader, model, DEVICE)
    print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    plt.ioff()
    plt.show()


