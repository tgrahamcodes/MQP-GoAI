# from game import *
import numpy as np
import sys
# from Players.mcts import *
# from Players.minimax import MiniMaxPlayer, GameState
from ..Players.linearregression import LinearRegression
#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_neural_net():
    # Set up batch size N, D_in is input,
    # H is hidden dimension , and D_out is output dimension
    N, D_in, H, D_out = 64, 10, 10, 10

    # Hardcoded numpy arrays to test model "normalized"
    x1 = np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.88])
    y1 = np.array([.4,.5,.6,.8,.10,.12,.14,.16,.18,.20])

    # Instatiating the model
    model = LinearRegression(D_in, H, D_out, x1, y1)
    assert (model != None)
#-------------------------------------------------------------------------

#-------------------------------------------------------------------------
def test_dict_split():
    N, D_in, H, D_out = 64, 10, 5, 2

    dict = {'0.2': 1, '0.3': 2, '0.4': 3, '0.5': 4, '0.6': 5}
    result = dict.items()
    data = list(result)
    n = np.array(data, dtype=np.float16)

    # Instatiating the model
    model = LinearRegression(D_in, H, D_out, n, n)

    assert (model != None)
#-------------------------------------------------------------------------
