import torch as th
import torch.nn.functional as F
#from problem1 import *

# -----------------------------------------------------------
# The class definition of Policy Network (PNet)
class PNet(th.nn.Module):
    def __init__(self):
        super(PNet, self).__init__()
        self.conv = th.nn.Conv2d(3, 20, 3) # convolutional layer 
        self.fc = th.nn.Linear(20, 9) # linear layer
    def forward(self, x):
        x = F.relu(self.conv(x)) # convolutional layer
        x = x.view(-1, 20) # flattening
        x = self.fc(x) # linear layer
        return x
# -----------------------------------------------------------
# The class definition of Player 1: Using only Policy Network to choose a move
# This network is trained with human supervision using labeled data
class PlayerPNet:
    def choose_a_move(self, s):
        pass
    def train(self, data_loader):
        pass

