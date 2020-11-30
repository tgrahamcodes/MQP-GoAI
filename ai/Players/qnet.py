#-------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------------------------
class QNet(ABC):

    @abstractmethod
    def __init__(self, size_in):
        pass

    @abstractmethod
    def forward(self, states):
        pass

    @abstractmethod
    def train(self, data_loader):
        pass