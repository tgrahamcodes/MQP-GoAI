#-------------------------------------------------------------------------
from abc import ABC, abstractmethod

#-------------------------------------------------------
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