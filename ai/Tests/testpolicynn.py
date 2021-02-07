import numpy as np
import sys
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from Players.minimax import RandomPlayer, MiniMaxPlayer, GameState
from Players.policynn import *
from game import Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Policy NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_adjust_logit():
    '''adjust_rewards'''
    #---------------------    
    g = TicTacToe()
    model = PolicyNN(g.input_size, g.out_size)
    x = torch.Tensor([np.zeros((9))])

    #---------------------
    b=np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn
    state = s.b.flatten().tolist()
    state.append(s.x)
    s = torch.Tensor([state])
    model.adjust_logit(s, x)
    adjusted = x.detach().numpy()[0]
    assert np.allclose(adjusted, np.zeros((9)))

    #---------------------
    x = torch.Tensor([np.zeros((9))])
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    state = s.b.flatten().tolist()
    state.append(s.x)
    s = torch.Tensor([state])
    model.adjust_logit(s, x)
    adjusted = x.detach().numpy()[0]
    assert np.allclose(adjusted, np.array([0, -1000, -1000, 0, -1000, -1000, 0, -1000, -1000]))

    #---------------------
    g = Othello()
    x = torch.Tensor([np.zeros((g.input_size-1))])
    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    state = s.b.flatten().tolist()
    state.append(s.x)
    s = torch.Tensor([state])
    model.adjust_logit(s, x)
    adjusted = x.detach().numpy()[0]
    assert np.allclose(adjusted, np.zeros((g.input_size-1)))

#-------------------------------------------------------------------------
def test_choose_a_move():
    '''choose_a_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = PolicyNNPlayer()
    assert p.file == None
    assert p.model == None

    #---------------------
    b=np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    p.model = PolicyNN(g.input_size, g.out_size)
    assert p.file == Path(__file__).parents[1].joinpath('Players/Memory/PolicyNN_TicTacToe.pt')
    assert type(p.model) == PolicyNN
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
        p.model = PolicyNN(g.input_size, g.out_size)
        r,c = p.choose_a_move(g,s)
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
    p1 = PolicyNNPlayer()
    p1.file = p1.select_file(g1)
    assert p1.file == Path(__file__).parents[1].joinpath('Players/Memory/PolicyNN_TicTacToe.pt')

    #---------------------
    g2 = Othello()
    p2 = PolicyNNPlayer()
    p2.file = p2.select_file(g2)
    assert p2.file == Path(__file__).parents[1].joinpath('Players/Memory/PolicyNN_Othello.pt')

    #---------------------
    g3 = GO(5)
    p3 = PolicyNNPlayer()
    p3.file = p3.select_file(g3)
    assert p3.file == Path(__file__).parents[1].joinpath('Players/Memory/PolicyNN_GO_5x5.pt')

    #---------------------
    g4 = GO(10)
    p4 = PolicyNNPlayer()
    p4.file = p4.select_file(g4)
    assert p4.file == Path(__file__).parents[1].joinpath('Players/Memory/PolicyNN_GO_10x10.pt')

#-------------------------------------------------------------------------
def test_export_model():
    '''export_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomPlayer()
    p1 = PolicyNNPlayer()

    #---------------------
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    _ = g.run_a_game(p1, p)
    assert Path.is_file(p1.file)

    # #---------------------
    # g = Othello()
    # p2 = PolicyNNPlayer()
    # b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0],
    #             [ 0, 0, 0, 0, 0, 0, 0, 0]])
    # s = GameState(b,x=1)
    # _ = g.run_a_game(p2, p)
    # assert Path.is_file(p2.file)

    # #---------------------
    # g = GO(5)
    # p3 = PolicyNNPlayer()
    # b=np.zeros((5,5))
    # s = GameState(b,x=1)
    # _ = g.run_a_game(p3, p)
    # assert Path.is_file(p3.file)

#-------------------------------------------------------------------------
def test_load():
    '''load'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = PolicyNNPlayer()
    p.load(g)
    assert p.file != None
    assert p.model != None
    assert type(p.model) == PolicyNN

    #---------------------
    # Game: Othello
    g = Othello()  # game
    p = PolicyNNPlayer()
    p.load(g)
    assert p.file != None
    assert p.model != None
    assert type(p.model) == PolicyNN

    #---------------------
    # Game: Go
    g = GO(5)  # game
    p = PolicyNNPlayer()
    p.load(g)
    assert p.file != None
    assert p.model != None
    assert type(p.model) == PolicyNN

#-------------------------------------------------------------------------
def test_train():
    '''train'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    model = PolicyNN(g.channels, g.N, g.output_size) 

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

    states = torch.Tensor([
        [player1,
        opponent1,
        empty1],
        [player2,
        opponent2,
        empty2],
    ])
    labels = [3, 3]

    class sample_data(Dataset):
        def __init__(self, states, labels):
            self.states = states
            self.labels = labels
        def __len__(self):
            return len(self.states) 
        def __getitem__(self, index):
            state = self.states[index]
            label = self.labels[index]
            return state, label
    d = sample_data(states, labels)
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    model.train(data_loader)

    # print values of forward function
    print('After training:')
    for states, labels in data_loader:
        outputs = model(states)
        print('Expected: ', [obj.item() for obj in labels])
        print('Output: ', [list(obj.detach().numpy()) for obj in outputs])
    
    assert False