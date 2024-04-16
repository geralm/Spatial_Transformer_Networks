import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

plt.ion()   # interactive mode
class Net(nn.Module):
    def __init__(self, channels:int=3, img_size:int=512, feature_scale:int=4, batch_size:int=32):
        super(Net, self).__init__()
        # channels: RGB = 3
        self.channels:int=channels 
        self.img_size:int=img_size
        self.batch_size:int=batch_size
        filters = [64, 128, 256, 512, 1024]
        self.filters = [int(x / feature_scale) for x in filters]

        self.conv1 = nn.Conv2d(self.channels, self.filters[0], kernel_size=5)
        self.conv2 = nn.Conv2d( self.filters[0], self.filters[1], kernel_size=5)
        self.conv2_drop = nn.Dropout2d()

        self.fc1 = nn.Linear(32*253*253, 50)
        self.fc2 = nn.Linear(50, 19)

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.Conv2d(self.channels, self.filters[0], kernel_size=7),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(self.filters[0], self.filters[1], kernel_size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(True)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32*252*252, self.filters[2]), 
            nn.ReLU(True),
            nn.Linear(self.filters[2], 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0] , dtype=torch.float))

    # Spatial transformer network forward function
    def stn(self, x):
        print(x.size())
        xs = self.localization(x)
        print(f"XS before  view {xs.size()}")
        batch_size = x.size(0)
        features = xs.size(1) * xs.size(2) * xs.size(3)
        xs = xs.view(batch_size, features)
        print(f"XS after  view {xs.size()}")
        theta = self.fc_loc(xs)

        print(f"Theta before  view {theta.size()}")
        theta = theta.view(-1, 2, 3)
        print(f"Theta after  view {theta.size()} vs {x.size()}")
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid,align_corners=False)

        return x

    def forward(self, x):
        # transform the input
        x = self.stn(x)

        # Perform the usual forward pass
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        print(f"Forward {x.size()}")
        x = x.view(x.size(0),32*253*253) 
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    def build(self, device):
        return self.to(device)

def train(device, train_loader, epoch, model, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        # optimizer.step()
        if batch_idx % 500 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), 0))
    #
    # A simple test procedure to measure the STN performances on MNIST.

def test(model, device, test_loader):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # sum up batch loss
            # test_loss += F.nll_loss(output, target, size_average=False).item()
            # # get the index of the max log-probability
            # pred = output.max(1, keepdim=True)[1]
            # correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))
def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

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

def run(device, train_data, test_data):
    model = Net().build(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    print(f"Training on {device}")
    for epoch in range(1, 3 + 1):
        train(device, train_data, epoch, model, optimizer)
        test(model, device, test_data)
    # Visualize the STN transformation on some input batch
    visualize_stn(test_data, model, device)

    plt.ioff()
    plt.show()