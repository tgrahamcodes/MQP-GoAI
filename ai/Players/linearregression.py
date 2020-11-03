# -------------------------------------------------------------------------
import torch
import os
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import numpy as np
# -------------------------------------------------------------------------
class LinearRegression(nn.Module):

    def __init__(self, D_in, H, D_out, x1, y1):
        super(LinearRegression, self).__init__()

        xt = np.array([x1])
        yt = np.array([y1])

        x = torch.from_numpy(xt)
        y = torch.from_numpy(yt)

        # Initializing the loss function
        loss_fn = nn.MSELoss()

        # Initializing the optimizer
        lr = 1e-4
        opt = torch.optim.SGD(model.parameters(), lr)

        epochs = 500
        losses=[]

        # Actually training the model
        for i in range(epochs):
            # Compute y by passing x to the model
            y_pred = model(x.float())

            loss = loss_fn(y_pred, y.float())
            if i % 100 == 99:
                print(i, loss.item())

        # Zero the gradients before the backwards pass
        model.zero_grad()

        # Compute the loss gradient
        loss.backward()

        # Update the weights using gradient descent
        with torch.no_grad():
            for i in model.parameters():
                i += lr * i.grad

        self.lin1 = nn.Linear(D_in, H)
        self.lin2 = nn.Linear(H, D_out)

    def forward(self, X):
        h_relu = self.lin1(X).clamp(min=0)
        pred = self.lin2(h_relu)
        return pred
# -------------------------------------------------------------------------
