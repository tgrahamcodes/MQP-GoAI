#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
from pathlib import Path
from .minimax import Node
from game import Player, GO, Othello, TicTacToe
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class RandomNN(nn.Module):

    def __init__(self, size_in):
        super().__init__()
        self.hidden = nn.Linear(size_in, size_in//2)
        self.output = nn.Linear(size_in//2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, s):
        state = s.b.flatten().tolist()
        state.append(s.x)
        x = torch.Tensor([state])
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.tanh(x)
        return x

#-------------------------------------------------------
class RandomNNPlayer(Player):

    def __init__(self):
        self.file = None
        self.model = None

    # ----------------------------------------------
    def choose_a_move(self,g,s):
        if not self.file:
            self.load_model(g)
        
        v = []
        c = []
        p=g.get_move_state_pairs(s)
        # expand the node with one level of children nodes 
        for m, s in p:
            # for each next move m, predict and append result
            tensor = self.model.forward(s)
            v.append(float(tensor.detach().numpy()[0,0]))
            c.append(m)
        # get index of max predicted value and return move
        idx = np.argmax(np.array(v))
        r,c = c[idx]
        return r,c
    
    # ----------------------------------------------
    def select_file(self, g):
        if isinstance(g, GO):
            return Path(__file__).parents[0].joinpath('Memory/RandomNN_' + g.__class__.__name__ + '_' + str(g.N) + 'x' + str(g.N) + '.pt')
        else:
            return Path(__file__).parents[0].joinpath('Memory/RandomNN_' + g.__class__.__name__ + '.pt')

    # ----------------------------------------------
    def export_model(self):
        torch.save(self.model, self.file)

    # ----------------------------------------------
    def load_model(self, g):
        self.file = self.select_file(g)
        if Path.is_file(self.file):
            self.model = torch.load(self.file)
        else:
            self.model = RandomNN(g.input_size)