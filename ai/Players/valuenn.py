#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
from math import sqrt
from pathlib import Path
from .vnet import VNet
from .minimax import Node, GameState
from game import Player, GO, GO_state, Othello, TicTacToe
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class ValueNN(VNet):

    #TODO: Input is three channel matrix stacked together, output is sum of value, one number
    def __init__(self, channels, N, output_size=1):
        # output size = 9, to switch to q network
        super().__init__()
        self.conv =  nn.Conv2d(channels, 20, N)
        # input size is 20 output size is 1
        self.output = nn.Linear(20, output_size)

    def forward(self, states):
        x = self.conv(states)
        # ReLU
        x = x.view(x.size(0), self.num_flat_features(x))
        x = self.output(x)
        # output 9 diff scores 
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train(self, data_loader, epochs=500):

        # Must use different loss funciton
        loss_fn = nn.MSELoss
        # MSELoss (supervised training)
        # define a func supervised training, then we have reinforced training
        # in rt we no longer have value
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        # s, a, reward
        # input game state -> get 9 scores then use output a1[i] to select action 
        # so only take one regression score, minimize distance between choice and reward
        
        for epoch in range(epochs):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, labels = mini_batch
                # labels will be values for this case
                optimizer.zero_grad()
                outputs = self(states)
                loss = loss_fn(outputs, labels)
                loss.backward() 
                optimizer.step()

                running_loss += loss.item()

                if i == (len(data_loader)-1):
                    print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))


#-------------------------------------------------------
class ValueNNPlayer(Player):

    # q function and v function, v takes 
    # gs- given start what is highest
    # q - takes 2 input game state (q(s,a)) best future reward
    # given s, a1, a2, a3, will return v1, v2, v3 
    # use the q function to get v
    # try each action and see which gives highest score
    # design q function - output a s

    # game state in, output 9, and choose highest score
    # 0 1 -1, only matters, -2 means taken
    # guarenteed to win = 1 
    # " " lose = -1
    # s, a=4, -1 (one training sample)
    # s, a=9, 1
    # s, a=8, 1
    # design a few to test the model
    # plug in state to get value to test   

    # after this reimplement refinforcement training, bellman equation
    # run a game s inital, a action, s after action completed, reward
    # in ML video lecture, 
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
            states = torch.Tensor([[player, opponent, empty, banned]])
        else:
            states = torch.Tensor([[player, opponent, empty]])

        tensor = self.model(states)
        v = tensor.detach().numpy()[0]
        idx = np.argmax(v)
        r,c = g.convert_index(idx)
        return r,c
    
    # ----------------------------------------------
    def select_file(self, g):
        if isinstance(g, GO):
            return Path(__file__).parents[0].joinpath('Memory/ValueNN_' + g.__class__.__name__ + '_' + str(g.N) + 'x' + str(g.N) + '.pt')
        else:
            return Path(__file__).parents[0].joinpath('Memory/ValueNN_' + g.__class__.__name__ + '.pt')

    # ----------------------------------------------
    def load(self, g):
        self.file = self.select_file(g)
        self.model = ValueNN(g.channels, g.N, g.output_size) 
        if Path.is_file(self.file):
            self.model.load_model(self.file)