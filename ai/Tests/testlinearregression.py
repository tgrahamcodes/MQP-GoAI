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
from pathlib import Path
#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_neural_net():  
    # Hardcoded numpy arrays to test model "normalized"
    x = torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y = torch.tensor([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])

    # Instatiating the model
    model = LinearRegression()
    model.train(x, y, 500, 0.01)

    # TODO find ground truth and compare it to what the model is predicting

    assert (model)
    assert (type(model) == LinearRegression)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_save():
    # Hardcoded numpy arrays to test model "normalized"
    x = torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y = torch.tensor([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])

    # Instatiating the model
    model = LinearRegression()
    model.train(x,y, 500, 0.01)
    model.save()
    assert (model)
    assert (type(model) == LinearRegression)
    assert Path(__file__).parents[1].joinpath('Players/Memory/MM_LinearReg.p')
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_data_load():
    # Hardcoded numpy arrays to test model "normalized"
    x = torch.tensor([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y = torch.tensor([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])

    # Instatiating the model
    model = LinearRegression()
    model.train(x,y, 500, 1e4)
    output = model.load()
    assert (model)
    assert (type(model) == LinearRegression)
    assert (output)
#-------------------------------------------------------------------------