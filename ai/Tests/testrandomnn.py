import numpy as np
import sys
import os
from pathlib import Path
from Players.minimax import RandomPlayer, MiniMaxPlayer, GameState
from Players.randomnn import *
from game import Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Random NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_choose_a_move():
    '''choose_a_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomNNPlayer()
    assert p.file == None
    assert p.model == None

    #---------------------
    b=np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    p.model = RandomNN(g.input_size)
    assert p.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_TicTacToe.pt')
    assert type(p.model) == RandomNN
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
        p.model = RandomNN(g.input_size)
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
    p1 = RandomNNPlayer()
    p1.file = p1.select_file(g1)
    assert p1.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_TicTacToe.pt')

    #---------------------
    g2 = Othello()
    p2 = RandomNNPlayer()
    p2.file = p2.select_file(g2)
    assert p2.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_Othello.pt')

    #---------------------
    g3 = GO(5)
    p3 = RandomNNPlayer()
    p3.file = p3.select_file(g3)
    assert p3.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_GO_5x5.pt')

    #---------------------
    g4 = GO(10)
    p4 = RandomNNPlayer()
    p4.file = p4.select_file(g4)
    assert p4.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_GO_10x10.pt')

#-------------------------------------------------------------------------
def test_export_model():
    '''export_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p1 = RandomNNPlayer()
    p2 = RandomPlayer()

    #---------------------
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    _ = g.run_a_game(p1, p2)
    assert Path.is_file(p1.file)

    #---------------------
    g = Othello()
    p3 = RandomNNPlayer()
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    _ = g.run_a_game(p3, p2)
    assert Path.is_file(p3.file)

    #---------------------
    g = GO(5)
    p4 = RandomNNPlayer()
    b=np.zeros((5,5))
    s = GameState(b,x=1)
    _ = g.run_a_game(p4, p2)
    assert Path.is_file(p4.file)

#-------------------------------------------------------------------------
def test_load_model():
    '''load_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomNNPlayer()
    p.load_model(g)
    assert p.file != None
    print(p.model, p.file)
    assert p.model != None
    assert type(p.model) == RandomNN

    #---------------------
    # Game: Othello
    g = Othello()  # game
    p = RandomNNPlayer()
    p.load_model(g)
    assert p.file != None
    assert p.model != None
    assert type(p.model) == RandomNN

    #---------------------
    # Game: Go
    g = GO(5)  # game
    p = RandomNNPlayer()
    p.load_model(g)
    assert p.file != None
    assert p.model != None
    assert type(p.model) == RandomNN

#-------------------------------------------------------------------------
def test_train():
    '''train'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    model = RandomNN(g.input_size) 

    #---------------------
    b=np.array([[0, 1, 1],
                [0,-1,-1],
                [0,-1, 1]])
    s=GameState(b,x=1) #it's X player's turn

    p=g.get_move_state_pairs(s)
    states = []
    for m, s in p:
        states.append(s)
    labels = [1, 0, -1]

    # train model
    model.train(states, labels)

    # print values of forward function
    print('After training:')
    for i, s in enumerate(states):
        tensor = model.forward(s)
        print('Expected: %d \t Output: %.3f' % (labels[i], float(tensor.detach().numpy()[0,0])))
        
    assert False
