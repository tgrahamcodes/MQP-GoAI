# -------------------------------------------------------------------------
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod
# -------------------------------------------------------------------------
class LinearRegression(nn.Module):

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.lr = 1e-4
        self.epochs = 500
        self.file = None
        self.lin = nn.Linear(10, 10)

    def forward(self, x):
        pred = self.lin(x)
        return pred

    def save(self):
        torch.save(self.lin, 'Players/Memory/MM_LinearReg.p')
        return True

    def load(self):
        torch.load('Players/Memory/MM_LinearReg.p')
        return True
        
    def train(self, x, y, epochs, lr):
        # Initializing the loss function
        loss_fn = torch.nn.MSELoss(reduction='sum')
        opt = torch.optim.SGD(self.parameters(), lr)

        # Actually training the model
        for i in range(epochs):
            # Compute y by passing x to the model
            y_pred = self.forward(x)

            loss = loss_fn(y_pred, y)
    
            # Zero the gradients before the backwards pass
            opt.zero_grad()

            # Compute the loss gradient
            loss.backward()

            opt.step()
            if (i % 100 == 99):
                print('epoch ', i, 'loss ', loss.item())

            with torch.no_grad():
                for param in self.parameters():
                    param -= lr * param.grad
# -------------------------------------------------------------------------
