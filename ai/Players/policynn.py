#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
from math import sqrt
from pathlib import Path
from .minimax import Node
from game import Player, GO, Othello, TicTacToe
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class PolicyNN(nn.Module):

    def __init__(self, size_in):
        super().__init__()
        self.output = nn.Linear(size_in, size_in-1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, s):
        state = s.b.flatten().tolist()
        state.append(s.x)
        x = torch.Tensor([state])
        x = self.output(x)
        x = self.relu(x)
        x = self.adjust_rewards(state, x)
        x = self.softmax(x)
        print('Probabilities:', list(x.detach().numpy()[0]))
        return x

    def adjust_rewards(self, state, x):
        # remove appended player's turn
        state.pop()
        # -1000 reward if move is invalid
        valid_moves = [-1000*abs(m) for m in state]
        current = x.detach().numpy()[0]
        adjusted = current + valid_moves
        print('Adjusted rewards:', list(adjusted))
        adjusted = torch.Tensor([adjusted])
        return adjusted


#-------------------------------------------------------
class PolicyNNPlayer(Player):

    def __init__(self):
        self.file = None
        self.model = None

    # ----------------------------------------------
    def choose_a_move(self,g,s):
        if not self.file:
            self.load_model(g)

        tensor = self.model.forward(s)
        p = tensor.detach().numpy()[0]
        idx = np.argmax(p)
        r = int(idx // (sqrt((g.input_size-1))))
        c = int(idx % (sqrt((g.input_size-1))))
        return r,c
    
    # ----------------------------------------------
    def select_file(self, g):
        if isinstance(g, GO):
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '_' + str(g.N) + 'x' + str(g.N) + '.pt')
        else:
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '.pt')

    # ----------------------------------------------
    def export_model(self):
        torch.save(self.model, self.file)

    # ----------------------------------------------
    def load_model(self, g):
        self.file = self.select_file(g)
        if Path.is_file(self.file):
            self.model = torch.load(self.file)
        else:
            self.model = PolicyNN(g.input_size)