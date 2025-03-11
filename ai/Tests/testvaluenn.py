import numpy as np
import sys
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ..Players.minimax import RandomPlayer, MiniMaxPlayer
from ..Players.valuenn import ValueNN, ValueNNPlayer
from ..game import GameState, Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Value NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_choose_a_move():
    '''choose_a_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()
    v = ValueNNPlayer()
    assert v.file == None
    assert v.model == None
    ep_greed = 0.9
    
    # -1 taken by Y
    # 1 taken by X
    # 0 free
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) # It is X player's turn
    
    r,c = v.choose_a_move(g,s,ep_greed) 
    count = 0
    for i in range(100):
        r,c = v.choose_a_move(g,s,ep_greed) 
        print ("Testing : (90%) Chose move: (r,c) ", r, c, ep_greed)

    v.model = ValueNN(g.channels, g.N, g.output_size)
    assert v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt')
    assert type(v.model) == ValueNN

    # Choose another move based on .10, so random
    ep_greed = 0.1
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) # It is X player's turn
    r,c = v.choose_a_move(g,s,ep_greed) 

    for i in range(100):
        r,c = v.choose_a_move(g,s,ep_greed) 
        print ("Exploring : (10%) Chose move: (r,c) ", r, c, ep_greed)
    
    assert False
#-------------------------------------------------------------------------
def test_select_file():
    '''select_file'''
    #---------------------
    # Game: TicTacToe
    g1 = TicTacToe()
    v1 = ValueNNPlayer()
    v1.file = v1.select_file(g1)
    assert v1.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt')

    #---------------------
    # Game: Othello
    g2 = Othello()
    v2 = ValueNNPlayer()
    v2.file = v2.select_file(g2)
    assert v2.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_Othello.pt')

    #---------------------
    # Game: Go
    # 5x5 Playing board
    g3 = GO(5)
    v3 = ValueNNPlayer()
    v3.file = v3.select_file(g3)
    assert v3.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_5x5.pt')

    #---------------------
    # Game: Go
    # 10x10 Playing board
    g4 = GO(10)
    v4 = ValueNNPlayer()
    v4.file = v4.select_file(g4)
    assert v4.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_10x10.pt')

#-------------------------------------------------------------------------
def test_save_model():
    '''save_model'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe() 
    p = RandomPlayer()
    v = ValueNNPlayer()

    b=np.array([[1, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn

    # Saving the model and then testing the model by comparing the filenames and type
    v.save(g)
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt'))
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN
    
    #---------------------
    # Game: Othello
    g = Othello()
    v = ValueNNPlayer()
    v.save(g)
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_Othello.pt'))
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN

    #---------------------
    # Game: Go
    g = GO(5)
    v = ValueNNPlayer()
    v.save(g)
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_5x5.pt'))
    assert v.file != None
    assert v.model != None
    assert type(v.model) == ValueNN

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
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_TicTacToe.pt'))
    assert type(v.model) == ValueNN

    #---------------------
    # Game: Othello
    g = Othello()  # game
    v = ValueNNPlayer()
    v.load(g)
    assert v.file != None
    assert v.model != None
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_Othello.pt'))
    assert type(v.model) == ValueNN

    #---------------------
    # Game: Go
    g = GO(5)  # game
    v = ValueNNPlayer()
    v.load(g)
    assert v.file != None
    assert v.model != None
    assert (v.file == Path(__file__).parents[1].joinpath('Players/Memory/ValueNN_GO_5x5.pt'))
    assert type(v.model) == ValueNN

#-------------------------------------------------------------------------
def test_train():
    '''train'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()
    v = ValueNN(g.channels, g.N, g.output_size) 

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

    # The sample data class to be provided to the training function which accepts a
    # Dataset object.
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
    data_loader = DataLoader(d, batch_size=1, shuffle=True, num_workers=0)
    v.train(data_loader)

    # Print out the training data from the forward function
    print('\nAfter training:\n')
    for states, rewards in data_loader:
        outputs = v(states)
        print('Expected output: ', [obj for obj in rewards])
        print('Actual output: ', [list(obj.detach().numpy()) for obj in outputs],'\n')

#-------------------------------------------------------------------------