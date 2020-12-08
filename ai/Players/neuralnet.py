#-------------------------------------------------------------------------
from abc import ABC, abstractmethod
from torch import nn
import torch

#-------------------------------------------------------------------------
class NeuralNet(ABC, nn.Module):

    def save_model(self, file):
        torch.save(self, file)

    def load_model(self, file):
        torch.load(file)

    @abstractmethod
    def train(self, data_loader, epochs):
        pass