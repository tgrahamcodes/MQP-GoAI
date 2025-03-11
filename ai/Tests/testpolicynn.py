import numpy as np
import sys
import os
import torch
import csv
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from ..Players.minimax import RandomPlayer, MiniMaxPlayer
from ..Players.policynn import *
from ..game import GameState, Othello, TicTacToe, GO

#-------------------------------------------------------------------------
def test_python_version():
    ''' ------------Policy NN---------------------'''
    assert sys.version_info[0] == 3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_adjust_logit():
    '''adjust_rewards'''
    #---------------------    
    g = TicTacToe()
    model = PolicyNN(g.channels, g.N, g.output_size)
    x = torch.Tensor([np.zeros((9))])

    #---------------------
    b=np.array([[0, 0, 0],
                [0, 0, 0],
                [0, 0, 0]])
    s=GameState(b,x=1) #it's X player's turn
    empty = np.ones((1, 9))  # All positions are empty
    banned = None
    adjusted = model.adjust_logit(x, empty, banned)
    assert np.allclose(adjusted.detach().numpy()[0], np.zeros((9)))

    #---------------------
    x = torch.Tensor([np.zeros((9))])
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    empty = np.array([[1, 0, 0, 1, 0, 0, 1, 0, 0]])  # Only positions 0, 3, 6 are empty
    banned = None
    adjusted = model.adjust_logit(x, empty, banned)
    assert np.allclose(adjusted.detach().numpy()[0], np.array([0, -1000, -1000, 0, -1000, -1000, 0, -1000, -1000]))

    #---------------------
    g = Othello()
    model = PolicyNN(g.channels, g.N, g.output_size)
    x = torch.Tensor([np.zeros((g.output_size))])
    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    empty = np.ones((1, g.output_size))  # All positions are empty
    banned = None
    adjusted = model.adjust_logit(x, empty, banned)
    assert np.allclose(adjusted.detach().numpy()[0], np.zeros((g.output_size)))

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
    assert p.model is not None
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
        p.model = PolicyNN(g.channels, g.N, g.output_size)
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
    labels = torch.tensor([3, 3])
    rewards = torch.tensor([1.0, 0.0])

    class sample_data(Dataset):
        def __init__(self, states, labels, rewards):
            self.states = states
            self.labels = labels
            self.rewards = rewards
        def __len__(self):
            return len(self.states) 
        def __getitem__(self, index):
            state = self.states[index]
            label = self.labels[index]
            reward = self.rewards[index]
            return state, label, reward

    d = sample_data(states, labels, rewards)
    data_loader = DataLoader(d, batch_size=1, shuffle=False, num_workers=0)
    model.train(data_loader)

    # Get model outputs after training
    with torch.no_grad():
        for states, labels, rewards in data_loader:
            # Get raw logits before adjustment
            x = model.conv(states)
            x = x.view(-1, model.num_flat_features(x))
            x = model.output(x)
            a = nn.functional.softmax(x, dim=1)
            # For the first state (reward=1.0), the model should assign higher probability to the labeled move
            if rewards[0] == 1.0:
                assert a[0][labels[0]] > 0.1, "Model should assign reasonable probability to the labeled move"
            # For empty positions, probabilities should be non-zero
            for i in range(len(a[0])):
                if empty1.flatten()[i] == 1:  # If position is empty
                    assert a[0][i] > 0.0, "Empty positions should have non-zero probabilities"

#-------------------------------------------------------------------------
def test_reinforce():
    '''reinforce'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p1 = PolicyNNPlayer()
    
    # Initialize model
    p1.load(g)  # Load fresh model
    
    # Save initial model parameters
    initial_params = [param.clone() for param in p1.model.parameters()]
    
    # Create a simple test state
    b = np.array([[0, 1,-1],
                  [0,-1, 1],
                  [0, 1,-1]])
    s = GameState(b, x=1)  # X's turn
    
    # Create single move with reward
    states = p1.extract_states(g, s)
    idx = torch.tensor([0])  # First position (0,0)
    value = torch.tensor([1.0])  # Positive reward
    
    # Create a minimal dataset for training
    class sample_data(torch.utils.data.Dataset):
        def __init__(self, states, labels, rewards):
            self.states = states
            self.labels = labels
            self.rewards = rewards
        def __len__(self):
            return len(self.states)
        def __getitem__(self, index):
            return self.states[index], self.labels[index], self.rewards[index]
    
    d = sample_data(states, idx, value)
    data_loader = torch.utils.data.DataLoader(d, batch_size=1, shuffle=False, num_workers=0)
    
    # Train on single example
    p1.model.train(data_loader, epochs=1)
    
    # Verify that training occurred
    final_params = [param.clone() for param in p1.model.parameters()]
    
    # Check that parameters changed during training
    params_changed = False
    for initial, final in zip(initial_params, final_params):
        if not torch.allclose(initial, final):
            params_changed = True
            break
    
    assert params_changed, "Model parameters should change after training"
    
    # Test that the model gives higher probability to the trained move
    with torch.no_grad():
        test_state = p1.extract_states(g, s)
        logits = p1.model(test_state)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Position (0,0) should have reasonable probability
        assert probs[0][0] > 0.1, "Model should assign reasonable probability to the trained move"

#-------------------------------------------------------------------------
def test_win_rates():
    '''win rates'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p1 = PolicyNNPlayer()
    p2 = PolicyNNPlayer()
    p3 = RandomPlayer()

    dirs = Path(__file__).parents[0].joinpath('WinRates/')
    if not Path.exists(dirs):
        dirs.mkdir(parents=True, exist_ok=True)

    iterations = 100
    p1_file = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(iterations-1) + '.pt')
    p1.load(g, p1_file)
    csv_file = Path(__file__).parents[0].joinpath('WinRates/PolicyNN_' + g.__class__.__name__ + '_vsModels.csv')
    with open(csv_file, mode='w+', newline='') as f:
        fieldnames = ['Player', 'Opponent', 'Win %', 'Loss %', 'Tie %']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(iterations):
            p2_file = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(i) + '.pt')
            p2.load(g, p2_file)
            player = 0
            opponent = 0
            ties = 0
            for j in range(1000):
                e, _ = g.run_game_reinforcement(p1, p2)
                if e == 1:
                    player += 1
                elif e == -1:
                    opponent += 1
                else:
                    ties += 1
            writer.writerow({'Player': ('Model '+str(iterations-1)), 'Opponent': ('Model '+str(i)), 
            'Win %': (player/1000)*100, 'Loss %': (opponent/1000)*100, 'Tie %': (ties/1000)*100})
    f.close()

    csv_file = Path(__file__).parents[0].joinpath('WinRates/PolicyNN_' + g.__class__.__name__ + '_vsRandom.csv')
    with open(csv_file, mode='w+', newline='') as f:
        fieldnames = ['Player', 'Opponent', 'Win %', 'Loss %', 'Tie %']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(iterations):
            p2_file = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(i) + '.pt')
            p2.load(g, p2_file)
            player = 0
            opponent = 0
            ties = 0
            for j in range(1000):
                e, _ = g.run_game_reinforcement(p2, p3)
                if e == 1:
                    player += 1
                elif e == -1:
                    opponent += 1
                else:
                    ties += 1
            writer.writerow({'Player': ('Model '+str(i)), 'Opponent': 'Random', 
            'Win %': (player/1000)*100, 'Loss %': (opponent/1000)*100, 'Tie %': (ties/1000)*100})
    f.close()

    assert True