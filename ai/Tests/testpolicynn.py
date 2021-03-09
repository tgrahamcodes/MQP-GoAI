import numpy as np
import sys
import os
import torch
import csv
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
    rewards = [1, 0]

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
    data_loader = DataLoader(d, batch_size=2, shuffle=False, num_workers=0)
    model.train(data_loader)

    # print values of forward function
    print('After training:')
    for states, labels, rewards in data_loader:
        z = model(states)
        a = nn.functional.softmax(z)
        print('Label: ', [obj.item() for obj in labels], '  Reward: ', [obj.item() for obj in rewards])
        print('Output: ', [list(obj.detach().numpy()) for obj in a])
    
    assert False

#-------------------------------------------------------------------------
def test_reinforce():
    '''reinforce'''
    #---------------------
    # Game: TicTacToe
    g = Othello()  # game
    p1 = PolicyNNPlayer()
    p2 = PolicyNNPlayer()

    #---------------------
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

    dirs = Path(__file__).parents[0].joinpath('Versions/')
    if not Path.exists(dirs):
        dirs.mkdir(parents=True, exist_ok=True)

    decay = 0.99
    iterations = 100
    matches = 1000
    for i in range(10):
        if i == 0:
            load_f = None
        else:
            load_f = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(i-1) + '.pt')
        p1.load(g, load_f)
        save_f = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(i) + '.pt')
        p1.set_file(save_f)

        if i >= 10:
            player_b_load = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(i-10) + '.pt')
            p2.load(g, player_b_load)

        state_tensor = torch.zeros((1, g.channels, g.N, g.N))
        state_list = []
        idxs = []
        values = []
        for k in range(10):
            e, moves = g.run_game_reinforcement(p1, p2)
            for j, move in enumerate(moves):
                s, r, c = move
                states = p1.extract_states(g, s)
                idx = r*g.N + c
                value = e * (decay**(len(moves)-j))
                if k == 0 and j == 0:
                    state_tensor = states
                else:
                    state_tensor = torch.cat((state_tensor, states))
                idxs.append(idx)
                values.append(value)
        d = sample_data(state_tensor, idxs, values)
        data_loader = DataLoader(d, batch_size=100, shuffle=True, num_workers=0)
        p1.model.train(data_loader)
        p1.model.save_model(p1.file)
        print("Model", i, "trained")

    assert False


#-------------------------------------------------------------------------
def test_win_rates():
    '''win rates'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game
    p1 = PolicyNNPlayer()
    p2 = PolicyNNPlayer()
    p3 = PolicyNNPlayer()

    dirs = Path(__file__).parents[0].joinpath('WinRates/')
    if not Path.exists(dirs):
        dirs.mkdir(parents=True, exist_ok=True)

    iterations = 100
    p1_file = Path(__file__).parents[0].joinpath('Versions/PolicyNN_' + g.__class__.__name__ + '_Version' + str(iterations-1) + '.pt')
    p1.load(g, p1_file)
    csv_file = Path(__file__).parents[0].joinpath('WinRates/PolicyNN_' + g.__class__.__name__ + '_vsModels.csv')
    with open(csv_file, mode='w+') as f:
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
    with open(csv_file, mode='w+') as f:
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