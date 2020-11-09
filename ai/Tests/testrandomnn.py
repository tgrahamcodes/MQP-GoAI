import numpy as np
import sys
import os
from pathlib import Path
from Players.minimax import MiniMaxPlayer, GameState
from Players.randomnn import *
from game import Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Random NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_expand_and_predict():
    '''expand_and_predict'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomNNPlayer()
    p.model = RandomNN(10)

    #---------------------
    # Current Node (root)
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    n = RNNNode(s) 
    n.expand_and_predict(g,p)
    assert type(n) == RNNNode
    assert n.s.x == 1
    assert np.allclose(n.s.b, b)
    assert len(n.c) == 3
    for c in n.c:
        assert type(c) == RNNNode
        assert c.p == n
        assert c.s.x == -1
        assert -1 <= c.v <= 1

    #---------------------
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    n = RNNNode(s) 
    n.expand_and_predict(g,p)
    assert type(n) == RNNNode
    assert n.s.x == -1
    assert np.allclose(n.s.b, b)
    assert len(n.c) == 8
    for c in n.c:
        assert type(c) == RNNNode
        assert c.p == n
        assert c.s.x == 1
        assert -1 <= c.v <= 1

#-------------------------------------------------------------------------
def test_choose_optimal_move():
    '''choose_optimal_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomNNPlayer()
    p.model = RandomNN(10)

    #---------------------
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    n = RNNNode(s) 
    n.expand_and_predict(g,p)
    v = [c.v for c in n.c]
    idx=np.argmax(np.array(v)*n.s.x)
    r1,c1 = n.c[idx].m
    r2,c2 = p.choose_optimal_move(n)
    assert r1 == r2
    assert c1 == c2

    #---------------------
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    n = RNNNode(s) 
    n.expand_and_predict(g,p)
    for c in n.c:
        c.v = 0.1
    r,c = p.choose_optimal_move(n)
    assert r == 0
    assert c == 0

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
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    assert p.file == Path(__file__).parents[1].joinpath('Players/Memory/RandomNN_TicTacToe.pt')
    assert type(p.model) == RandomNN
    assert r in {0,1,2}
    assert c in {0,1,2}

    #---------------------
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    assert r in {0,1,2}
    assert c == 0

    #---------------------
    b=np.array([[ 0, 0, 0],
                [-1,-1, 1],
                [ 1, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    assert r == 0
    assert c in {0,1,2}

    #---------------------
    b=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 1, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    r,c = p.choose_a_move(g,s)
    assert r == 0
    assert c == 0

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

#-------------------------------------------------------------------------
def test_export_model():
    '''export_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p1 = RandomNNPlayer()
    p2 = RandomNNPlayer()

    #---------------------
    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=-1) #it's X player's turn
    _ = g.run_a_game(p1, p2)
    assert Path.is_file(p1.file)

#-------------------------------------------------------------------------
def test_load_model():
    '''load_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p = RandomNNPlayer()
    p.file = p.select_file(g)
    p.model = p.load_model()
    assert type(p.model) == RandomNN
    assert p.model != None
