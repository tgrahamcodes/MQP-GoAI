import numpy as np
import sys
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Players.minimax import RandomPlayer, MiniMaxPlayer, GameState
from Players.valuenn import ValueNN, ValueNNPlayer
from game import Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Value NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_choose_a_move():
    '''choose_a_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    v = ValueNNPlayer()
    assert v.file == None
    assert v.model == None

    #---------------------
    b=np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = v.choose_a_move(g,s)
    v.model = ValueNN(g.channels, g.N, g.output_size)
    assert v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt')
    assert type(v.model) == ValueNN
    assert r in {0,1,2}
    assert c in {0,1,2}

    #---------------------
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn

    m0 = 0
    m1 = 0
    m2 = 0
    for _ in range(100):
        v.model = ValueNN(g.channels, g.N, g.output_size)
        r,c = v.choose_a_move(g,s)
        if (r,c) == (0,0): m0 += 1
        if (r,c) == (1,0): m1 += 1
        if (r,c) == (2,0): m2 += 1
    assert m0 < 50
    assert m1 < 50
    assert m2 < 50

#-------------------------------------------------------------------------
def test_select_file():
    '''select_file'''
    #---------------------
    g1 = TicTacToe()
    v1 = ValueNNPlayer()
    v1.file = v1.select_file(g1)
    assert v1.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt')

    #---------------------
    g2 = Othello()
    v2 = ValueNNPlayer()
    v2.file = v2.select_file(g2)
    assert v2.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_Othello.pt')

    #---------------------
    g3 = GO(5)
    v3 = ValueNNPlayer()
    v3.file = v3.select_file(g3)
    assert v3.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_5x5.pt')

    #---------------------
    g4 = GO(10)
    v4 = ValueNNPlayer()
    v4.file = v4.select_file(g4)
    assert v4.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_10x10.pt')

#-------------------------------------------------------------------------
def test_export_model():
    '''export_model'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe() 
    p = RandomPlayer()
    v1 = ValueNN(g.channels, g.N)

    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])

    # b current board state
    # x current user playing (-1 for O && 1 for x)

    s=GameState(b,x=-1)

    # X Player(v1) vs O Player(p)
    _ = g.run_a_game(v1, p)
    assert Path.is_file(v1.file)

    #---------------------
    # Game: Othello
    g = Othello()
    v2 = ValueNN()
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    _ = g.run_a_game(v3, p)
    assert Path.is_file(p3.file)

    #---------------------
    # Game: GO
    g = GO(5)
    v3 = ValueNN()
    b=np.zeros((5,5))
    s = GameState(b,x=1)
    _ = g.run_a_game(v3, p)
    assert Path.is_file(v3.file)

#-------------------------------------------------------------------------
def test_load():
    '''load'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    v = ValueNNPlayer()
    v.load(g)
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN

    #---------------------
    # Game: Othello
    g = Othello()  # game
    v = ValueNNPlayer()
    v.load(g)
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN

    #---------------------
    # Game: Go
    g = GO(5)  # game
    v = ValueNNPlayer()
    v.load(g)
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN

#-------------------------------------------------------------------------
def test_train():
    '''train'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    v = ValueNN(g.channels, g.N, g.output_size) 
    vp = ValueNNPlayer()
    #---------------------
    # [ 0, 1, 1]
    # [ 0,-1,-1]
    # [ 0, 0, 1]
    # x = -1
    player1 = np.array([0, 0, 0, 0, 1, 1, 0, 0, 0]).reshape(3,3)
    opponent1 = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1]).reshape(3,3)
    empty1 = np.array([1, 0, 0, 1, 0, 0, 1, 1, 0]).reshape(3,3)

    # [-1, 1, 1]
    # [ 0,-1,-1]
    # [ 0, 0, 1]
    # x = 1
    player2 = np.array([0, 1, 1, 0, 0, 0, 0, 0, 1]).reshape(3,3)
    opponent2 = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0]).reshape(3,3)
    empty2 = np.array([0, 0, 0, 1, 0, 0, 1, 1, 0]).reshape(3,3)

    states = torch.Tensor([[player1, opponent1, empty1], [player2, opponent2, empty2],])
    rewards = torch.Tensor([[0,-1,-1,0,0,-1,1,1,0],[0,0,1,0,0,-1,1,0,0]])

    class sample_data(Dataset):
        def __init__(self, states, rewards):
            self.states = states
            self.rewards = rewards
        def __len__(self):
            return len(self.states) 
        def __getitem__(self, index):
            state = self.states[index]
            reward = self.rewards[index]
            return state, reward
    d = sample_data(states, rewards)
    data_loader = DataLoader(d, batch_size=1, shuffle=False, num_workers=0)
    v.train(data_loader)

    # print values of forward function
    print('After training:')
    for states, rewards in data_loader:
        outputs = v(states)
        print('Expected: ', [obj for obj in rewards])
        print('Output: ', [list(obj.detach().numpy()) for obj in outputs])
    
    assert False