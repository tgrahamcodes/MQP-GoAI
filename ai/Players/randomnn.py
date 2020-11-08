#-------------------------------------------------------------------------
from torch import nn
import numpy as np
import torch
from pathlib import Path
from .minimax import Node
from game import Player
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
class RandomNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(10, 5)
        self.output = nn.Linear(5, 1)
        self.tanh = nn.Tanh()

    def forward(self, n):
        state = n.s.b.flatten().tolist()
        state.append(n.s.x)
        x = torch.Tensor([state])
        x = self.hidden(x)
        x = self.tanh(x)
        x = self.output(x)
        x = self.tanh(x)
        return x

# ----------------------------------------------
class RNNNode(Node):

    def __init__(self,s,p=None,c=None,m=None,v=None):
        super(RNNNode, self).__init__(s,p=p,c=c,m=m,v=v)

    # ----------------------------------------------
    def expand_and_predict(self,g,player):
        # get the list of valid next move-state pairs from the current game state
        p=g.get_move_state_pairs(self.s)
         
        # expand the node with one level of children nodes 
        for m, s in p:
            # for each next move m and game state s, create a child node
            c = RNNNode(s,p=self, m=m)
            tensor = player.model.forward(c)
            c.v = float(tensor.detach().numpy()[0,0])
            # append the child node the child list of the current node 
            self.c.append(c)
        #########################################

#-------------------------------------------------------
class RandomNNPlayer(Player):

    def __init__(self):
        self.model = RandomNN()

    #----------------------------------------------
    def choose_optimal_move(self,n):
        v = [c.v for c in n.c]
        idx=np.argmax(np.array(v)*n.s.x)
        r,c = n.c[idx].m
        return r,c

    # ----------------------------------------------
    def choose_a_move(self,g,s):        
        n = RNNNode(s)
        n.expand_and_predict(g,self)
        r,c = self.choose_optimal_move(n)
        return r,c
