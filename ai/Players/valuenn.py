#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
import math
import random
from pathlib import Path
from .neuralnet import NeuralNet
from .minimax import Node, GameState
from game import Player, GO, GO_state, Othello, TicTacToe
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class ValueNN(NeuralNet):

    def __init__(self, channels, N, output_size=1):
        super().__init__()
        self.conv =  nn.Conv2d(channels, 20, N)
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Linear(20, output_size)
    
    def forward(self, states):
        x = self.conv(states)
        x = self.relu(x)
        x = x.view(x.size(0), self.num_flat_features(x))
        x = self.output(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def adjust_logit(self, x, empty, banned):
        for i in range(len(empty[0])):
            if empty[0][i] == 0 or (banned and banned[0][i] == 1):
                x[0][i] -= 1000 
        return x

    def supervised_learning(states, matrix, location, opp_location, empty_location):
        #For each game state, 3 channel matrix, our location, opponents location, empty location
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(epochs):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, actions, rewards = mini_batch
                optimizer.zero_grad()
                
                outputs = self(states)
                loss = loss_fn(outputs, rewards)
                loss.backward() 
                
                optimizer.step()

                running_loss += loss.item()
                
                if (states == rewards):
                    print('True')
        

                if i == (len(data_loader)-1):
                    print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))


    def reinforced_training(data_loader):
        pass

    def train(self, data_loader, epochs=500):
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr=0.001, momentum=0.9)
        
        for epoch in range(epochs):
            running_loss = 0.0

            for i, mini_batch in enumerate(data_loader):
                states, rewards = mini_batch
                optimizer.zero_grad()
                
                outputs = self(states)
                loss = loss_fn(outputs, rewards)
                loss.backward() 
                
                optimizer.step()

                running_loss += loss.item()

                if i == (len(data_loader)-1):
                    print('epoch %d: %.3f' % (epoch+1, running_loss/len(data_loader)))

#-------------------------------------------------------
class ValueNNPlayer(Player):

    # ----------------------------------------------
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

        v = []

        states = self.extract_states(g, s)
        z = self.model(states)
 
        r = random.random()
        v = z.detach().numpy()
        
        if (r >= 0.9):
            m = g.get_valid_moves(s)
            idx = random.randint(0, len(m)-1)
            r,c = m[idx]
        else:
            idx = np.argmax(np.array(v))
            r,c = g.convert_index(idx)        

        return r, c
    
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

    # ----------------------------------------------
    def save(self, g):
        self.file = self.select_file(g)
        self.model = ValueNN(g.channels, g.N, g.output_size) 
        if Path.is_file(self.file):
            self.model.save_model(self.file)