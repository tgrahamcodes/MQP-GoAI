# -------------------------------------------------------------------------
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
from torch.autograd import Variable
# -------------------------------------------------------------------------
class LinearRegression(nn.Module):

    def __init__(self, input, output):
        super(LinearRegression, self).__init__()
        self.lr = 1e-4
        self.epochs = 500
        self.file = None
        self.lin = nn.Linear(input, output)

    def forward(self, x):
        pred = self.lin(x)
        return pred

    def save(self, file):
        torch.save(self.lin.state_dict(), file)

    def load(self, file):
        self.lin.load_state_dict(torch.load(file))
        self.lin.eval()

    def train(self, data_loader, epochs=500):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, labels = mini_batch
                optimizer.zero_grad()
                outputs = self(states)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i == (len(data_loader)-1):
                    print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))

    # def train(self, x, y):
    #     # Initializing the loss function
       
    #     loss_fn = torch.nn.MSELoss(reduction='sum')
    #     opt = torch.optim.SGD(self.parameters(), self.lr)

    #     # Actually training the model
    #     for epoch in range(self.epochs):
    #         # Compute y by passing x to the model

    #         y_pred = self.forward(x)

    #         loss = loss_fn(y_pred, y)
    
    #         # Zero the gradients before the backwards pass
    #         opt.zero_grad()

    #         # Compute the loss gradient
    #         loss.backward()

    #         opt.step()
    #         print('epoch ', epoch, 'loss ', loss.item())

    #         with torch.no_grad():
    #             for param in self.parameters():
    #                 param -= self.lr * param.grad
# -------------------------------------------------------------------------
