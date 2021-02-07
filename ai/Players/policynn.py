#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
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
        x = self.conv(states)
        x = x.view(-1, self.num_flat_features(x))
        x = self.output(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def adjust_logit(self, states, x):
        return

    def train(self, data_loader, epochs=500):
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
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
            self.load(g)

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

        tensor = self.model(states)
        p = tensor.detach().numpy()[0]
        idx = np.argmax(p)
        r,c = g.convert_index(idx)
        return r,c
    
    # ----------------------------------------------
    def select_file(self, g):
        if isinstance(g, GO):
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '_' + str(g.N) + 'x' + str(g.N) + '.pt')
        else:
            return Path(__file__).parents[0].joinpath('Memory/PolicyNN_' + g.__class__.__name__ + '.pt')

    # ----------------------------------------------
    def load(self, g):
        self.file = self.select_file(g)
        self.model = PolicyNN(g.channels, g.N, g.output_size) 
        if Path.is_file(self.file):
            self.model.load_model(self.file)