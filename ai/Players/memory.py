#-------------------------------------------------------------------------
from abc import ABC, abstractmethod
from torch import nn
import torch

#-------------------------------------------------------------------------
class MemoryDict(ABC):

    def get_node(self, s):
        return self.dictionary.get(s, None)

    @abstractmethod
    def fill_mem(self, n):
        pass

    @abstractmethod
    def export_mem(self, file):
        pass
    
    @abstractmethod
    def load_mem(self, file):
        pass

#-------------------------------------------------------------------------
class RandomNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, n):
        state = n.s.b.flatten().tolist()
        state.append(n.s.x)
        x = torch.Tensor([state])
        x = self.hidden(x)
        x = self.sigmoid(x)
        x = self.output(x)
        return x