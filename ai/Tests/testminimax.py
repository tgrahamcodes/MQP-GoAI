import copy
import numpy as np
import sys
import os
sys.path.append(os.path.abspath('..\\GoAI\\ai\\Players'))
from minimax import *
sys.path.append(os.path.abspath('..\\GoAI\\ai'))
from game import GameState, TicTacToe, Othello, GO_state, GO

'''
    Unit test 1:
    This file includes unit tests for minimax.py.
    You could test the correctness of your code by typing `nosetests -v testminimax.py` in the terminal.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Random and Minimax--------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)


#-------------------------------------------------------------------------
def test_get_valid_moves():
    '''get_valid_moves()'''
    g = TicTacToe()  # game 

    b=np.array([[  1 , 0 ,-1 ],
                [ -1 , 1 , 0 ],
                [  1 , 0 ,-1 ]])
    s= GameState(b,x=1)
    m=g.get_valid_moves(s)
    assert type(m)==list
    assert len(m)==3
    for i in m:
        assert i== (0,1) or i== (1,2) or i == (2,1)
    assert m[0]!=m[1] and m[1]!=m[2]

    
    s= GameState(np.zeros((3,3)),x=1)
    m=g.get_valid_moves(s)
    assert len(m)==9

    b=np.array([[  1 , 0 ,-1 ],
                [  0 , 0 , 0 ],
                [  1 , 0 ,-1 ]])
    s= GameState(b,x=1)
    m=g.get_valid_moves(s)
    assert len(m)==5
    for i in m:
        assert i== (0,1) or i== (1,0) or i== (1,1) or i== (1,2) or i== (2,1)  


#-------------------------------------------------------------------------
def test_check_game():
    '''check_game()'''
    g = TicTacToe()  # game 
    b=np.array([[ 1, 0, 1],
                [ 0, 0,-1],
                [ 0,-1, 0]])
    s= GameState(b,x=1)
    e = g.check_game(s)
    assert e is None # the game has not ended yet

    b=np.array([[ 1,-1, 1],
                [ 0, 1,-1],
                [-1, 1,-1]])
    s= GameState(b,x=1)
    e = g.check_game(s)
    assert e is None # the game has not ended yet

    b=np.array([[ 1, 1, 1],
                [ 0, 0,-1],
                [ 0,-1, 0]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins

    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[-1, 0, 0],
                [ 1, 1, 1],
                [ 0,-1, 0]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[-1, 0, 0],
                [ 0, 0,-1],
                [ 1, 1, 1]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins
    
    b=np.array([[ 1, 0, 0],
                [ 1, 0,-1],
                [ 1,-1, 0]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[-1, 1, 0 ],
                [ 0, 1, 0 ],
                [-1, 1, 0 ]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[-1, 0, 1],
                [ 0, 0, 1],
                [-1, 0, 1]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[ 1, 0, 0],
                [ 0, 1,-1],
                [-1, 0, 1]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins

    b=np.array([[-1, 0, 1],
                [ 0, 1, 0],
                [ 1, 0,-1]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 1  # x player wins
    s= GameState(-b,x=1)
    e = g.check_game(s)
    assert e ==-1  # O player wins


    b=np.array([[-1, 1,-1],
                [ 1, 1,-1],
                [ 1,-1, 1]])
    s= GameState(b,x=-1)
    e = g.check_game(s)
    assert e == 0 # a tie






#-------------------------------------------------------------------------
def test_choose_a_move():
    '''random choose_a_move()'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 
    p = RandomPlayer()
    b=np.array([[ 0, 1, 1],
                [ 1, 0,-1],
                [ 1, 1, 0]])

    b_=np.array([[ 0, 1, 1],
                 [ 1, 0,-1],
                 [ 1, 1, 0]])
    s= GameState(b,x=1)
    count=np.zeros(3)
    for _ in range(100):
        r,c = p.choose_a_move(g,s)
        assert b_[r,c]==0 # player needs to choose a valid move 
        assert np.allclose(s.b,b_) # the player should never change the game state object
        assert r==c # in this example the valid moves are on the diagonal of the matrix
        assert r>-1 and r<3
        count[c]+=1
    assert count[0]>20 # the random player should give roughly equal chance to each valid move
    assert count[1]>20
    assert count[2]>20
    
    b=np.array([[ 1, 1, 0],
                [ 1, 0,-1],
                [ 0, 1, 1]])

    s= GameState(b,x=1)
    for _ in range(100):
        r,c = p.choose_a_move(g,s)
        assert b[r,c]==0 
        assert r==2-c 
        assert r>-1 and r<3


    #---------------------
    # The AI agent should also be compatible with the game Othello.
    # now let's test on the game "Othello":
    g = Othello()  # game 
    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1,-1,-1, 0, 0],
                [ 0, 0, 0, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    p = RandomPlayer()
    s= GameState(b,x=1)
    count = np.zeros(5)
    for _ in range(200):
        r,c = p.choose_a_move(g,s)
        assert np.allclose(b,b_) # the player should never change the game state object
        assert b[r,c]==0 # player needs to choose a valid move 
        assert r==2 
        assert c>1 and c<7
        count[c-2]+=1
    assert count[0]>20 # the random player should give roughly equal chance to each valid move
    assert count[1]>20
    assert count[2]>20
    assert count[3]>20
    assert count[4]>20



    # test whether we can run a game using random player
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s= GameState(b,x=1)
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        assert e==-1



    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s= GameState(b,x=1)
    w=0
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        w+=e
    assert np.abs(w)<9


    # test whether we can run a game using random player
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s= GameState(b,x=1)
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        assert e==-1



    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    w=0
    s= GameState(b,x=1)
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        w+=e
    assert np.abs(w)<9

    #---------------------
    # The AI agent should also be compatible with the game: GO 
    # now let's test on the game "GO":
    g = GO(board_size=2)  # game (2 x 2 board)
    s = g.initial_game_state()
    p = RandomPlayer()

    b_= s.b.copy()
    count = np.zeros(5)
    for _ in range(200):
        r,c = p.choose_a_move(g,s)
        assert np.allclose(s.b,b_) # the player should never change the game state object
        assert s.a==0 
        if r is None and c is None: # the player choose to pass without placing any stone in the step
            count[-1]+=1
        else:
            count[2*r+c] +=1
    assert count[0]>20 # the random player should give roughly equal chance to each valid move
    assert count[1]>20
    assert count[2]>20
    assert count[3]>20
    assert count[4]>20


#-------------------------------------------------------------------------
def test_expand():
    '''expand'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    # Current Node (root)
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s = GameState(b,x=1) #it's X player's turn
    n = MMNode(s) 
    # expand
    n.expand(g)
    assert len(n.c) ==3 
    assert n.s.x==1
    b_=np.array([[0, 1,-1],
                 [0,-1, 1],
                 [0, 1,-1]])
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MMNode
        assert c.s.x==-1
        assert c.p==n
        assert c.c==[] #only add one level of children nodes, not two levels.
        assert c.v==None

    # child node A
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
    assert c

    # child node B
    b=np.array([[ 0, 1,-1],
                [ 1,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(1,0)
    assert c

    # child node C
    b=np.array([[ 0, 1,-1],
                [ 0,-1, 1],
                [ 1, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,0)
    assert c

    #--------------------------

    # Current Node (root)
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    s=GameState(b,x=-1) #it's O player's turn
    n = MMNode(s) 
    n.expand(g)
    assert n.s.x==-1
    assert len(n.c) ==2
    for c in n.c:
        assert c.s.x==1
        assert c.p==n
        assert c.c==[]

    # child node A
    b=np.array([[ 1, 1,-1],
                [-1,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(1,0)
    assert c

    # child node B
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [-1, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,0)
    assert c

    #---------------------------
    s=GameState(np.zeros((3,3)),x=1) #it's X player's turn
    n = MMNode(s)
    n.expand(g)
    assert n.s.x==1
    assert len(n.c) ==9
    for c in n.c:
        assert c.s.x==-1
        assert c.p==n
        assert c.c==[]
        assert np.sum(c.s.b)==1
        assert c.v==None

    #---------------------
    # The AI agent should also be compatible with Othello game.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_=b.copy()
    s=GameState(b,x=1) #it's X player's turn
    n = MMNode(s) 
    # expand
    n.expand(g)
    assert len(n.c) ==2 
    assert n.s.x==1
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MMNode
        assert c.p==n
        assert c.c==[]
        assert c.v==None

    # child node A
    b=np.array([[ 1, 1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
            assert x.s.x==1 # it is still X player's turn because there is no valid move for O player
    assert c

    # child node B
    b=np.array([[ 0,-1, 1, 1, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,4)
            assert x.s.x==-1 
    assert c


    
    #---------------------
    b=np.array([[ 0, 1,-1, 1, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_=b.copy()
    s=GameState(b,x=-1) #it's O player's turn
    n = MMNode(s) 
    # expand
    n.expand(g)
    print(n.c)
    assert len(n.c) ==3 
    assert n.s.x==-1
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MMNode
        assert c.p==n
        assert c.c==[]
        assert c.v==None

    # child node A
    b=np.array([[-1,-1,-1, 1, 0, 0, 0, 0],
                [ 0, 0, 1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
            assert x.s.x == -1 # no valid move for X player
    assert c

    # child node B
    b=np.array([[ 0, 1,-1,-1,-1, 0, 0, 0],
                [ 0, 0, 1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,4)
            assert x.s.x == 1 
    assert c

    # child node C
    b=np.array([[ 0, 1,-1, 1, 0, 0, 0, 0],
                [ 0, 0,-1, 0, 0, 0, 0, 0],
                [ 0, 0,-1, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,2)
            assert x.s.x == 1 
    assert c


    #---------------------
    # The AI agent should also be compatible with the game: GO 
    # now let's test on the game "GO":
    g = GO(board_size=2)  # game (2 x 2 board)
    b=np.array([[ 0, 1],
                [ 1, 0]])
    b_=b.copy()
    s=GO_state(b,x=-1) #it's O player's turn
    n = MMNode(s) 
    # expand
    n.expand(g)
    assert len(n.c) ==1 # only one valid move for O player: 'pass'
    assert n.s.x==-1
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    c = n.c[0]
    assert type(c)==MMNode
    assert c.p==n
    assert c.c==[]
    assert c.v==None
    assert np.allclose(c.s.b,b_)
    assert c.m[0] is None
    assert c.m[1] is None

    s=GO_state(b,x=1) #it's X player's turn
    n = MMNode(s) 
    # expand
    n.expand(g)
    assert len(n.c) ==3  


#-------------------------------------------------------------------------
def test_build_tree():
    '''build_tree'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    # current node (root node)
    b=np.array([[ 0, 1,-1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)

    # the current game state should not change after building the tree 
    assert np.allclose(b,b_)
    assert len(n.c) ==3 
    assert n.s.x==1
    assert n.v==None
    assert n.p==None
    assert n.m==None

    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MMNode
        assert c.s.x==-1
        assert c.p==n
        assert len(c.c)==2
        assert c.v==None

    #-----------------------
    # child node A
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
            ca=x
    assert c

    # child node B
    b=np.array([[ 0, 1,-1],
                [ 1,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(1,0)
            cb=x
    assert c

    # child node C
    b=np.array([[ 0, 1,-1],
                [ 0,-1, 1],
                [ 1, 1,-1]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,0)
            cc=x
    assert c

    #-----------------------
    # Child Node A's children
    for c in ca.c:
        assert c.s.x==1
        assert c.p==ca
        assert c.v==None

    # grand child node A1
    b=np.array([[ 1, 1,-1],
                [-1,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in ca.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(1,0)
            assert len(x.c)==1
            #-----------------------
            # Great Grand Child Node A11
            assert x.c[0].s.x==-1
            assert x.c[0].p==x
            assert x.c[0].v==None
            assert x.c[0].c==[]
    assert c

    # grand child node A2
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [-1, 1,-1]])
    c = False
    for x in ca.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,0)
            assert x.c== []
    assert c
    
    #-----------------------
    # Child Node B's children
    for c in cb.c:
        assert c.s.x==1
        assert c.p==cb
        assert c.c==[]
        assert c.v==None

    # grand child node B1
    b=np.array([[-1, 1,-1],
                [ 1,-1, 1],
                [ 0, 1,-1]])
    c = False
    for x in cb.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
    assert c

    # grand child node B2
    b=np.array([[ 0, 1,-1],
                [ 1,-1, 1],
                [-1, 1,-1]])
    c = False
    for x in cb.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(2,0)
    assert c

    #-----------------------
    # Child Node C's children
    for c in cc.c:
        assert c.s.x==1
        assert c.p==cc
        assert c.v==None

    # grand child node C1
    b=np.array([[-1, 1,-1],
                [ 0,-1, 1],
                [ 1, 1,-1]])
    c = False
    for x in cc.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
            assert x.c== []
    assert c

    # grand child node C2
    b=np.array([[ 0, 1,-1],
                [-1,-1, 1],
                [ 1, 1,-1]])
    c = False
    for x in cc.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(1,0)
            assert len(x.c)==1
            # Great Grand Child Node C21
            assert x.c[0].s.x==-1
            assert x.c[0].p==x
            assert x.c[0].v==None
            assert x.c[0].c==[]
    assert c


    #-----------------------
    b=np.array([[ 0, 0, 1],
                [ 0, 1, 1],
                [-1, 0,-1]])
    s = GameState(b,x=-1) #it's O player's turn
    n = MMNode(s) 
    n.build_tree(g)

    assert len(n.c) ==4 
    assert n.s.x==-1
    assert n.v==None
    assert n.p==None
    assert n.m==None
    
    b1=np.array([[-1, 0, 1],
                 [ 0, 1, 1],
                 [-1, 0,-1]])
    b2=np.array([[ 0,-1, 1],
                 [ 0, 1, 1],
                 [-1, 0,-1]])
    b3=np.array([[ 0, 0, 1],
                 [-1, 1, 1],
                 [-1, 0,-1]])
    b4=np.array([[ 0, 0, 1],
                 [ 0, 1, 1],
                 [-1,-1,-1]])

    for c in n.c:
        assert c.s.x== 1
        assert c.v==None
        assert c.p==n
        if np.allclose(c.s.b,b1):
            assert c.m == (0,0)
            assert len(c.c) ==3
        if np.allclose(c.s.b,b2):
            assert c.m == (0,1)
            assert len(c.c) ==3
        if np.allclose(c.s.b,b3):
            assert c.m == (1,0)
            assert len(c.c) ==3
        if np.allclose(c.s.b,b4):
            assert c.m == (2,1)
            assert c.c == [] #terminal node, no child


    # The AI agent should be compatible with both games: TicTacToe and Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)

    # the current game state should not change after building the tree 
    assert np.allclose(n.s.b,b_)
    assert len(n.c) ==2 
    assert n.s.x==1
    assert n.v==None
    assert n.p==None
    assert n.m==None

    for c in n.c:
        assert type(c)==MMNode
        assert c.s.x==-1
        assert c.p==n
        assert c.v==None
        assert len(c.c)==1
    #-----------------------
    # child node A
    b=np.array([[ 0, 0,-1, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,5)
            ca=x
    assert c

    #-----------------------
    # child node B
    b=np.array([[ 0, 1, 1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,1)
            cb=x
    assert c

    #-----------------------
    # Child Node A's children
    # grand child node A1
    assert ca.c[0].p==ca
    assert ca.c[0].v==None
    assert ca.c[0].m==(0,6)
    assert ca.c[0].c==[]
    b=np.array([[ 0, 0,-1,-1,-1,-1,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(ca.c[0].s.b,b)

    #-----------------------
    # Child Node B's children
    # grand child node B1
    assert cb.c[0].p==cb
    assert cb.c[0].v==None
    assert cb.c[0].m==(0,0)
    assert cb.c[0].c==[]
    b=np.array([[-1,-1,-1,-1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(cb.c[0].s.b,b)


    #------------------------------------
    b=np.array([[ 0,-1, 1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)

    # the current game state should not change after building the tree 
    assert np.allclose(n.s.b,b_)
    assert len(n.c) ==2 
    assert n.s.x==1
    assert n.v==None
    assert n.p==None
    assert n.m==None

    for c in n.c:
        assert type(c)==MMNode
        assert c.p==n
        assert c.v==None
        assert len(c.c)==1
    #-----------------------
    # child node A
    b=np.array([[ 1, 1, 1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,0)
            assert x.s.x==1 # there is no valid move for O player, so O player needs to give up the chance
            ca=x
    assert c

    #-----------------------
    # child node B
    b=np.array([[ 0,-1, 1, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    c = False
    for x in n.c:
        if np.allclose(x.s.b,b):
            c=True
            assert x.m==(0,5)
            assert x.s.x==-1
            cb=x
    assert c

    #-----------------------
    # Child Node A's children
    # grand child node A1
    assert ca.c[0].p==ca
    assert ca.c[0].v==None
    assert ca.c[0].m==(0,5)
    assert ca.c[0].c==[]
    b=np.array([[ 1, 1, 1, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(ca.c[0].s.b,b)

    #-----------------------
    # Child Node B's children
    # grand child node B1
    assert cb.c[0].p==cb
    assert cb.c[0].v==None
    assert cb.c[0].m==(0,6)
    assert cb.c[0].c==[]
    b=np.array([[ 0,-1,-1,-1,-1,-1,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(cb.c[0].s.b,b)

    #---------------------
    # The AI agent should also be compatible with the game: GO 
    # now let's test on the game "GO":
    g = GO(board_size=2)  # game (2 x 2 board)
    b=np.array([[ 1, 1],
                [ 1, 0]])
    s=GO_state(b,x=1,a=1) #it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    assert len(n.c) ==1 
    assert n.c[0].s.x==-1
    assert n.c[0].s.a ==2 
    assert len(n.c[0].c)==0

    s=GO_state(b,x=-1,p=(1,1),a=1) #it's O player's turn
    n = MMNode(s) 
    n.build_tree(g)
    assert len(n.c) ==1 
    assert n.c[0].s.x==1
    assert n.c[0].s.a ==2 
    assert len(n.c[0].c)==0

    g = GO(board_size=2,max_game_length=1)  # game (2 x 2 board)
    b=np.array([[ 1, 1],
                [ 1, 0]])
    s=GO_state(b,x=-1) #it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    assert len(n.c) ==2 
    assert n.c[0].s.x==1
    assert len(n.c[0].c)==0

    g = GO(board_size=2,max_game_length=2)  # game (2 x 2 board)
    b=np.array([[ 1, 1],
                [ 1, 0]])
    s=GO_state(b,x=-1) #it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    assert len(n.c) ==2 
    for c in n.c:
        assert c.s.x==1
        if np.allclose(c.s.b, b):
            assert len(c.c) == 1 
        else:
            assert len(c.c) == 4 

#-------------------------------------------------------------------------
def test_compute_v():
    '''compute_v()'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    #-------------------------
    # the value of a terminal node is its game result
    b=np.array([[ 1, 0, 0],
                [ 0, 1,-1],
                [ 0,-1, 1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.compute_v(g) 
    assert  n.v== 1 # X player won the game

    # the value of a terminal node is its game result
    b=np.array([[ 1, 1,-1],
                [-1, 1, 1],
                [ 1,-1,-1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.compute_v(g) 
    assert  n.v== 0 # tie 

    # the value of a terminal node is its game result
    b=np.array([[ 1, 0, 1],
                [ 0, 0, 1],
                [-1,-1,-1]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.compute_v(g) 
    assert  n.v==-1 # O player won the game

    #-------------------------
    # if it is X player's turn, the value of the current node is the max value of all its children nodes.

    b=np.array([[ 0,-1, 1],
                [ 0, 1,-1],
                [ 0,-1, 1]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    # the current node has 3 children nodes, two of which are terminal nodes (X player wins)
    n.compute_v(g) 
    # so the max value among the three children nodes max(1,?,1) = 1 (here ? is either 1 or 0 or -1)
    assert  n.v== 1 # X player won the game

    #-------------------------
    # if it is O player's turn, the value of the current node is the min value of all its children nodes.

    b=np.array([[ 0, 1,-1],
                [ 0,-1, 1],
                [ 1, 1,-1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.build_tree(g)
    # the current node has 2 children nodes, one of them is a terminal node (O player wins)
    n.compute_v(g) 
    # so the min value among the two children nodes min(-1,0) =-1 
    assert  n.v==-1 # O player won the game


    #-------------------------
    # a tie after one move
    b=np.array([[-1, 1,-1],
                [-1, 1, 1],
                [ 0,-1, 1]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 0  


    #-------------------------
    # optimal moves lead to: O player wins
    b=np.array([[-1, 1,-1],
                [ 1, 0, 0],
                [ 1, 0, 0]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v==-1

    #-------------------------
    # optimal moves lead to a tie
    b=np.array([[ 0, 1, 0],
                [ 0, 1, 0],
                [ 0, 0,-1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 0

    #-------------------------
    # optimal moves lead to: X player wins
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0,-1, 0]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1

    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 0,-1]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1

    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 0, 0]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1

    b=np.array([[ 1,-1, 1],
                [-1, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1

    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [-1, 0, 0]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1


    b=np.array([[ 1,-1, 1],
                [ 0, 0, 1],
                [ 0, 0,-1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v==-1

    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 1

    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 1,-1]])
    s = GameState(b,x=-1)
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    assert  n.v== 0

    # The AI agent should be compatible with both games: TicTacToe and Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    n.compute_v(g)
    assert np.allclose(n.s.b,b_)
    assert  n.v== 1
    
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    n.compute_v(g)
    assert  n.v==-1
    

    b=np.array([[ 0, 0,-1, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=-1) # it's O player's turn
    n = MMNode(s) 
    n.build_tree(g)
    n.compute_v(g)
    assert  n.v==-1
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    n.compute_v(g)
    assert  n.v==1


    b=np.array([[ 0,-1, 1,-1, 1,-1, 0, 0],
                [ 1, 0, 0, 0, 0, 0, 0, 0],
                [ 1, 0, 0, 0, 0, 0, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s) 
    n.build_tree(g)
    n.compute_v(g)
    assert  n.v==1


#-------------------------------------------------------------------------
def test_choose_optimal_move():
    '''choose_optimal_move()'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 
    p = MiniMaxPlayer()

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    r,c=p.choose_optimal_move(n)
    assert r == 2
    assert c == 0

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 1,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=-1) # it's O player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    r,c=p.choose_optimal_move(n)
    assert r == 2
    assert c == 0

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=-1) # it's O player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    r,c=p.choose_optimal_move(n)
    assert r == 1
    assert c == 1

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 1,-1],
                [-1, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g) 
    r,c=p.choose_optimal_move(n)
    assert r == 1
    assert c == 0


    # The AI agent should also be compatible with Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g)
    assert np.allclose(n.s.b,b_)
    r,c=p.choose_optimal_move(n)
    assert r == 0
    assert c == 0


#-------------------------------------------------------------------------
def test_minmax_choose_a_move():
    '''minmax choose_a_move()'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    # two possible moves: one leads to win
    p = MiniMaxPlayer()
    b=np.array([[ 0,-1, 1],
                [-1, 1, 1],
                [ 0, 1,-1]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    r, c = p.choose_a_move(g,s)
    assert np.allclose(s.b,b_)
    assert r==2  
    assert c==0  


    # three possible moves, one leads to win
    # p = MiniMaxPlayer()
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    r, c = p.choose_a_move(g,s) 
    assert r==2  
    assert c==0  

    #-------------------------
    # p = MiniMaxPlayer()
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=-1) # it's O player's turn
    r, c = p.choose_a_move(g,s) 
    assert r == 1
    assert c == 1


    #-------------------------
    # play against random player in the game
    # p1 = MiniMaxPlayer()
    p2 = RandomPlayer()

    # X Player: MinMax
    # O Player: Random 
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 0,-1]])
    for i in range(10):
        s = GameState(b,x=1) # it's X player's turn
        e = g.run_a_game(p,p2,s=s)
        assert e==1


    #-------------------------
    # play against MinMax player in the game

    # X Player: MinMax 
    # O Player: MinMax  
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    for i in range(10):
        s = GameState(b,x=1) # it's X player's turn
        e = g.run_a_game(p,p,s=s)
        assert e==1

    b=np.array([[ 0, 0, 1],
                [ 0,-1, 0],
                [ 1,-1, 0]])
    s = GameState(b,x=1) # it's X player's turn
    e = g.run_a_game(p,p,s=s)
    assert e==0

    b=np.array([[ 0, 0, 0],
                [ 0,-1, 0],
                [ 1, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    e = g.run_a_game(p,p,s=s)
    assert e==0

    b=np.array([[ 0, 0, 0],
                [ 0, 0, 0],
                [ 1,-1, 0]])
    s = GameState(b,x=1) # it's X player's turn
    e = g.run_a_game(p,p,s=s)
    assert e==1

    b=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 0]])
    s = GameState(b,x=1) # it's X player's turn
    e = g.run_a_game(p,p,s)
    assert e==1

    b=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [-1, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    e = g.run_a_game(p,p,s)
    assert e==0

    #******************************************************
    #*******************(TRY ME)***************************
    #******************************************************
    '''Run A Complete Game (TicTacToe): 
       the following code will run a complete TicTacToe game using MiniMaxPlayer,     
       if you want to try this, uncomment the following three lines of code.
       Note: it may take 1 or 2 minutes to run
    '''
    #g = TicTacToe()
    #e = g.run_a_game(p1,p1)
    #assert e==0
    #******************************************************
    #******************************************************
    #******************************************************



    #----------------------------------------------
    # The AI agent should be compatible with both games: TicTacToe and Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    for i in range(10):
        s = GameState(b,x=1) # it's X player's turn
        e = g.run_a_game(p,p2,s=s)
        assert e==1

    w=0
    for i in range(10):
        s = GameState(b,x=1) # it's X player's turn
        e = g.run_a_game(p2,p2,s=s)
        w+=e 
    assert np.abs(w)<9


    #******************************************************
    #*******************(DO NOT TRY ME:)*******************
    #******************************************************
    ''' Run A Complete Game (Othello): 
        The following code will run a complete Othello game using MiniMaxPlayer,     
        If you want to try this, uncomment the following two lines of code. 
        My suggestion: Don't let it run for a long time, stop the program in the terminal after 1 or 2 minutes.
        Otherwise it will eventually use up all your computer's memory. Even with unlimited memory, your computer will still need to run forever to build the search tree! So spending 1 or 2 minutes on this program should be enough to prove that there is no hope for MiniMax method on large board games. 
    '''
    #g = Othello()
    #e = g.run_a_game(p1,p1)
    #******************************************************
    #******************************************************
    #******************************************************


#-------------------------------------------------------------------------
def test_fill_mem():
    '''fill_mem'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 
    m = MMMemory()
    # fill dictionary with dummy value so it does not load from memory to test recursive part
    m.dictionary = {"temp" : "temp"}

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g)
    m.fill_mem(n)
    mem_n = m.dictionary.get(s, None)
    assert len(m.dictionary) > 0
    assert n == mem_n
    assert len(n.c) == 3

    #-------------------------
    p = MiniMaxPlayer()
    p.choose_a_move(g,s)
    n = p.mem.dictionary.get(s, None)
    assert isinstance(p.mem, MMMemory)
    assert len(p.mem.dictionary) > 0
    assert n != None

    #-------------------------
    p = MiniMaxPlayer()
    p.choose_a_move(g,s)
    n = MMNode(s)
    n.build_tree(g)
    mem_n = p.mem.dictionary.get(s, None)
    assert isinstance(p.mem, MMMemory)
    assert len(p.mem.dictionary) > 0
    assert [c.m for c in n.c] == [c.m for c in n.c]


#-------------------------------------------------------------------------
def test_get_node():
    '''get_node'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 
    m = MMMemory()
    m.dictionary = {"temp" : "temp"}

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    mem_n = m.get_node(s)
    assert mem_n == None
    n.build_tree(g)
    n.compute_v(g)
    m.fill_mem(n)
    mem_n = m.get_node(s)
    assert m.dictionary != {}
    assert n == mem_n


#-------------------------------------------------------------------------
def test_minimax_memory():
    '''minimax memory'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 
    p = MiniMaxPlayer()
    p.mem.dictionary = {"temp" : "temp"}

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g)
    p.mem.fill_mem(n) 
    mem_n = p.mem.get_node(s)
    assert p.mem.dictionary != {}
    assert n == mem_n

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 1,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=-1) # it's O player's turn
    n = p.mem.get_node(s)
    assert n != None

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 1,-1]])
    s = GameState(b,x=-1) # it's O player's turn
    n = p.mem.get_node(s)
    assert n == None
    _ = p.choose_a_move(g,s)
    n = p.mem.get_node(s)
    assert n != None

    #-------------------------
    b=np.array([[ 1,-1, 0],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    _ = p.choose_a_move(g,s)

    #-------------------------
    b=np.array([[ 1,-1, 0],
                [ 0, 1,-1],
                [ 0, 0, 1]])
    s = GameState(b,x=-1) # it's O player's turn
    n = p.mem.get_node(s)
    assert n != None
    assert n.c == []

    #-------------------------
    b=np.array([[ 1,-1, 0],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=1) # it's X player's turn
    n = p.mem.get_node(s)
    assert n != None


    # The AI agent should also be compatible with Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    b_ = b.copy()
    s = GameState(b,x=1) # it's X player's turn
    n = MMNode(s)
    n.build_tree(g)
    n.compute_v(g)
    p.mem.fill_mem(n)
    mem_n = p.mem.get_node(s)
    assert p.mem.dictionary != {}
    assert n == mem_n
    assert n.c != []

