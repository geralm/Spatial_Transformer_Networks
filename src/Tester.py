import torch
from utils import compute_accuracy
import numpy as np
import matplotlib.pyplot as plt
import torchvision
class Tester(): 
    def __init__(self, test_loader, model, device) -> None:
        self.model = model
        self.device = device
        self.test_loader = test_loader
    def random_test(self): 
        for batch_idx, (features, targets) in enumerate(self.test_loader):
            features = features
            targets = targets
            break
        self.model.eval()
        logits, probas = self.model(features.to(self.device)[0, None])
        print('Probability Female %.2f%%' % (probas[0][0]*100))
    def _convert_image_np(self,inp):
        """Convert a Tensor to numpy image."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp
    def visualize_stn(self):
        plt.ion()
        with torch.no_grad():
        # Get a batch of training data
            data = next(iter(self.test_loader))[0].to(self.device)

            input_tensor = data.cpu()
            transformed_input_tensor = self.model.stn(data).cpu()

            in_grid = self._convert_image_np(
                torchvision.utils.make_grid(input_tensor))

            out_grid = self._convert_image_np(
                torchvision.utils.make_grid(transformed_input_tensor))

            # Plot the results side-by-side
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(in_grid)
            axarr[0].set_title('Dataset Images')

            axarr[1].imshow(out_grid)
            axarr[1].set_title('Transformed Images')
        plt.ioff()
        plt.show()
    def run(self):
        self.model.eval()
        with torch.no_grad(): 
            print('Test accuracy: %.2f%%' % (compute_accuracy(self.model,self.test_loader, device=self.device)))
     