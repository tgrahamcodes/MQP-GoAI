#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
from torch.distributions import Categorical
from math import sqrt
from pathlib import Path
from .pnet import PNet
from .minimax import Node, GameState
from game import Player, GO, GO_state, Othello, TicTacToe
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class PolicyNN(PNet):

    def __init__(self, channels, N, output_size):
        super().__init__()
        self.conv =  nn.Conv2d(channels, 20, N)
        self.output = nn.Linear(20, output_size)

    def forward(self, states):
        empty = np.resize(torch.empty_like(states[0][2]).copy_(states[0][2]).detach().numpy(), (1, self.output.out_features))
        banned = np.resize(torch.empty_like(states[0][3]).copy_(states[0][3]).detach().numpy(), (1, self.output.out_features)) if self.conv.in_channels > 3 else None
        x = self.conv(states)
        x = x.view(-1, self.num_flat_features(x))
        x = self.output(x)
        x = self.adjust_logit(x, empty, banned)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def adjust_logit(self, x, empty, banned):
        for i in range(len(empty[0])):
            if empty[0][i] == 0 or (banned and banned[0][i] == 1):
                x[0][i] -= 1000 
        return x

    def train(self, data_loader, epochs=10):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, labels, rewards = mini_batch
                optimizer.zero_grad()
                z = self(states)
                a = nn.functional.softmax(z)
                x = Categorical(a)
                m = x.sample()
                logp = x.log_prob(m)
                loss = torch.sum(-logp * rewards)
                loss.backward()
                optimizer.step()

                # running_loss += loss.item()

                # if i == (len(data_loader)-1):
                #     print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))


#-------------------------------------------------------
class PolicyNNPlayer(Player):

    def __init__(self):
        self.file = None
        self.model = None

    # ----------------------------------------------
    def extract_states(self, g, s):
        player = np.zeros_like(s.b)
        opponent = np.zeros_like(s.b)
        empty = np.zeros_like(s.b)
        for i, row in enumerate(s.b):
            player[i] = [1 if x == s.x else 0 for x in row]
            opponent[i] = [1 if x == -s.x else 0 for x in row]
            empty[i] = [1 if x == 0 else 0 for x in row]

        if isinstance(g, GO):
            banned = np.zeros_like(s.b)
            if s.p:
                banned[pos[0]][pos[1]] = 1
            states = torch.Tensor([
                [player,
                opponent,
                empty,
                banned]
            ])
        else:
            states = torch.Tensor([
                [player,
                opponent,
                empty]
            ])

        return states

    # ----------------------------------------------
    def choose_a_move(self,g,s):
        if not self.file:
            self.load(g)

        states = self.extract_states(g, s)
        z = self.model(states)
        a = nn.functional.softmax(z)
        x = Categorical(a)
        m = x.sample()
        idx = m.item()
        r,c = g.convert_index(idx)
        return r,c
    
    # ----------------------------------------------
    def select_file(self, g):
        if isinstance(g, GO):
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '_' + str(g.N) + 'x' + str(g.N) + '.pt')
        else:
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '.pt')

    # ----------------------------------------------
    def set_file(self, file):
        self.file = file

    # ----------------------------------------------
    def load(self, g, file=None):
        self.file = self.select_file(g) if not file else file
        self.model = PolicyNN(g.channels, g.N, g.output_size) 
        if Path.is_file(self.file):
            self.model.load_model(self.file)