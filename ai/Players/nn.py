# -------------------------------------------------------------------------
import torch
import pickle
import os
# from Players.minimax import MMNode
# from Players.mcts import M
# from Players.memory import MemoryDict
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# -------------------------------------------------------------------------
class LinearRegression(nn.Module):
    
    def __init__(self,in_size,out_size):
        super().__init__()

        self.lin = nn.Linear(in_features=in_size, out_features=out_size)

    def forward(self, X):
        pred = self.lin(X)
        return(pred)

# -------------------------------------------------------------------------

[w,b] = model.parameters()

def get_parameters():
    return(w[0][0].item(),b[0].item())

# Hardcoded csv for practice
filename = Path("Players/testing.csv")
data = pd.read_csv(filename)
normalized = (data-data.min())/(data.max()-data.min())

# Spliting into training and testing
train, test = train_test_split(normalized, test_size=0.2)

# Converting training data into Tensors
X_train = torch.Tensor([[x] for x in list(train.Squaremeters)])
y_train = torch.FloatTensor([[y] for y in list(train.Prices)])

X_test = torch.Tensor([[x] for x in list(train.Squaremeters)])
y_test = torch.FloatTensor([[y] for y in list(train.Prices)])

torch.manual_seed(1)

# Initializing the model
model = LinearRegression(1,1)

# Initializing the loss function
loss_fn = nn.MSELoss()

# Initializing the optimizer
opt = torch.optim.SGD(model.parameters(), lr=0.1)

epochs = 200
losses=[]

# Actually training the model
for i in range(epochs):
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    print("Epochs: ",i, "Loss:", loss.item())
    opt.step()
    opt.zero_grad()
    losses.append(loss)