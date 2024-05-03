import torch
import time
from utils import compute_accuracy
class Trainer(): 
    def __init__(self, train_loader, validation_loader, model, device, settings) -> None:
        self._SEED = settings["model"]["RANDOM_SEED"]
        torch.manual_seed(self._SEED)
        self._LEARNING_RATE = settings["model"]["LEARNING_RATE"]
        self._NUM_EPOCHS =  settings["model"]["NUM_EPOCHS"]
        self._DEVICE = device

        self.train_loader = train_loader
        self.validation_loader = validation_loader

        self.model = model
        self.optimizer = torch.optim.SGD(model.parameters(), lr=self._LEARNING_RATE) 
        #self.optimizer =torch.optim.Adam(model.parameters(), lr=self._LEARNING_RATE) 
         
    def _trainfunc(self, epoch):
        self.model.train()
        cost_fn = torch.nn.CrossEntropyLoss()
        for batch_idx, (features,targets) in enumerate(self.train_loader):
            features, targets = features.to(self._DEVICE), targets.to(self._DEVICE)

            ### FORWARD AND BACK PROP
            logits, probas = self.model(features)
            cost = cost_fn(logits, targets)
            self.optimizer.zero_grad()     
            cost.backward()
            ### UPDATE MODEL PARAMETERS
            self.optimizer.step()
            ### LOGGING
            if not batch_idx % 50:
                print ('Epoch: %03d/%03d | Batch %04d/%04d | Cost: %.4f' 
                    %(epoch+1,self._NUM_EPOCHS , batch_idx, 
                        len(self.train_loader), cost))
    
    def _validatefunc(self, epoch): 
        self.model.eval()
        with torch.no_grad(): # save memory during inference
            print('Epoch: %03d/%03d | Train: %.3f%% | Valid: %.3f%%' % (
                        epoch+1, self._NUM_EPOCHS, 
                        compute_accuracy(self.model, self.train_loader, device=self._DEVICE),
                        compute_accuracy(self.model, self.validation_loader, device=self._DEVICE)))
    def run(self):
        start_time = time.time()
        for epoch in range(self._NUM_EPOCHS):
            self._trainfunc(epoch)
            self._validatefunc(epoch)
            print('Time elapsed: %.2f min' % ((time.time() - start_time)/60))
        print('Total Training Time: %.2f min' % ((time.time() - start_time)/60))
    
        return self.model
