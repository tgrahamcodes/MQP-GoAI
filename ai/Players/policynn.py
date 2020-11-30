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

    def forward(self, states):
        x = self.output(states)
        x = self.relu(x)
        self.adjust_rewards(states, x)
        x = self.softmax(x)
        # tmp = x.clone()
        # print('   Probabilities:', [list(obj.detach().numpy()) for obj in tmp])
        return x

    def adjust_rewards(self, states, x):
        for i, state in enumerate(states):
            # remove appended player's turn
            state = state[:-1]
            # -1000 reward if move is invalid
            valid_moves = torch.Tensor([-1000*abs(m) for m in state])
            current = x[i]
            x[i] = torch.add(current, valid_moves)
            # tmp = x[i].clone()
            # print('Adjusted rewards:', list(tmp.detach().numpy()))

    def train(self, data_loader):
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(500):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, labels = mini_batch
                optimizer.zero_grad()
                outputs = self(states)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                if i == (len(data_loader)-1):
                    print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))


#-------------------------------------------------------
class PolicyNNPlayer(Player):

    def __init__(self):
        self.file = None
        self.model = None

    # ----------------------------------------------
    def choose_a_move(self,g,s):
        if not self.file:
            self.load_model(g)

        tensor = self.model(s)
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