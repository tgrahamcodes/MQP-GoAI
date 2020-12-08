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

    def __init__(self, size_in, size_out):
        super().__init__()
        self.size_in = size_in
        self.size_out = size_out
        self.output = nn.Linear(size_in, size_out)

    def forward(self, states):
        x = self.output(states)
        self.adjust_logit(states, x)
        return x

    def adjust_logit(self, states, x):
        for i, state in enumerate(states):
            # m,s = self.get_valid_moves(state, g)
            # remove appended player's turn
            state = state[:(self.size_out-self.size_in)]
            # -1000 reward if move is invalid
            valid_moves = torch.Tensor([-1000*abs(m) for m in state])
            current = x[i]
            x[i] = torch.add(current, valid_moves)

    # def get_valid_moves(self, state, g):
    #     state_copy = state
    #     state_copy = state_copy.detach().numpy()
    #     state_info = state_copy[(self.size_out-self.size_in):].copy()
    #     dim = int(sqrt(g.out_size))
    #     board = np.reshape(state_copy[:(self.size_out-self.size_in)], (dim, dim))
    #     if len(state_info) == 4:
    #         p = None 
    #         p=g.get_move_state_pairs(GO_state(board, state_info[0], p, state_info[2], state_info[3]))
    #     else:
    #         p=g.get_move_state_pairs(GameState(board, state_info[0]))
    #     return p[0],p[1]

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

        state = s.b.flatten().tolist()
        state.append(s.x)
        if isinstance(g, GO):
            state.append(0 if s.p == None else len(s.p))
            state.append(s.a)
            state.append(s.t)
        s = torch.Tensor([state])
        tensor = self.model(s)
        p = tensor.detach().numpy()[0]
        idx = np.argmax(p)
        if isinstance(g, GO):
            r = int(idx // g.N)
            c = int(idx % g.N)
        else:
            r = int(idx // (sqrt((g.out_size))))
            c = int(idx % (sqrt((g.out_size))))
        print(r, c)
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
        self.model = PolicyNN(g.input_size, g.out_size)
        if Path.is_file(self.file):
            self.model.load_model(self.file)