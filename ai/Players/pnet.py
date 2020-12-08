#-------------------------------------------------------------------------
from abc import ABC, abstractmethod
from .neuralnet import NeuralNet

#-------------------------------------------------------------------------
class PNet(NeuralNet):

    @abstractmethod
    def adjust_logit(self, states, x):
        pass