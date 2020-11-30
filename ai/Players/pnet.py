#-------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------
class PNet(ABC):

    @abstractmethod
    def __init__(self, size_in):
        pass

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def train(self, data_loader):
        pass

    @abstractmethod
    def adjust_rewards(self, states, x):
        pass