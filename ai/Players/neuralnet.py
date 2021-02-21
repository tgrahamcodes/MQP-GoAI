#-------------------------------------------------------------------------
from abc import ABC, abstractmethod
from torch import nn
import torch

#-------------------------------------------------------------------------
class NeuralNet(ABC, nn.Module):

    def save_model(self, file):
        torch.save(self.state_dict(), file)

    def load_model(self, file):
        self.load_state_dict(torch.load(file))
        # self.train(True)

    # def save_model(self, file):
    #     torch.save(self, file)

    # def load_model(self, file):
    #     torch.load(file)

    @abstractmethod
    def train(self, data_loader, epochs):
        pass