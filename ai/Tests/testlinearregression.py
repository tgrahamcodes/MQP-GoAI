# from game import *
import numpy as np
import sys
# from Players.mcts import *
# from Players.minimax import MiniMaxPlayer, GameState
import torch.nn as nn
import torch
from ..Players.linearregression import LinearRegression
import os.path
from os import path
#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_neural_net():  
    # Hardcoded numpy arrays to test model "normalized"
    data=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    # Instatiating the model
    model = LinearRegression()
    model.train(data, data, 500, 0.01)
    
    assert (model != None)
    assert (type(model) == LinearRegression)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_save():
    # Hardcoded numpy arrays to test model "normalized"
    x1 = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y1 = np.array([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])
    xt = np.array([x1])
    yt = np.array([y1])

    x = torch.from_numpy(xt)
    y = torch.from_numpy(yt)

    # Instatiating the model
    model = LinearRegression()
    model.train(x,y, 500, 1e4)
    model.save()
    assert (model != None)
    assert (type(model) == LinearRegression)
    assert (path.isfile("../Memory/LinearReg.p"))
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_load():
    # Hardcoded numpy arrays to test model "normalized"
    x1 = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y1 = np.array([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])
    xt = np.array([x1])
    yt = np.array([y1])

    x = torch.from_numpy(xt)
    y = torch.from_numpy(yt)

    # Instatiating the model
    model = LinearRegression()
    model.train(x,y, 500, 1e4)
    output = model.load()
    assert (model != None)
    assert (type(model) == LinearRegression)
    assert (output)
#-------------------------------------------------------------------------