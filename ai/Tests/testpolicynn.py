import numpy as np
import sys
import os
from pathlib import Path
from Players.minimax import RandomPlayer, MiniMaxPlayer, GameState
from Players.policynn import *
from game import Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Policy NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)