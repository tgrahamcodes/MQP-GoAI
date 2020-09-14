from problem2 import *
from problem1 import MiniMaxPlayer, TicTacToe, GameState
from game import Othello 
import numpy as np
import sys

'''
    Unit test 2:
    This file includes unit tests for problem2.py.
'''

#-------------------------------------------------------------------------
def test_python_version():
    ''' ----------- Problem 2 (50 points in total)---------------------'''
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_sample():
    '''(5 points) sample'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    #------------------------
    b=np.array([ [ 0, 1, 1],
                 [ 0,-1, 1],
                 [-1, 1,-1]])
    bc=np.array([[ 0, 1, 1],
                 [ 0,-1, 1],
                 [-1, 1,-1]])
    s = GameState(b,x=-1) # "O" player's turn
    n = MCNode(s) 
    assert np.allclose(n.s.b,bc) # the game state should not change after simulation
    v=0
    for _ in range(100):
        e=n.sample(g)
        assert e==-1 or e==1
        v+=e
    assert np.abs(v)<25  # the two results should have roughly the same chance

    #------------------------
    b=np.array([[ 0, 1, 1],
                [-1,-1, 1],
                [-1, 1,-1]])
    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s) 
    for _ in range(100):
        e=n.sample(g)
        assert e==1

    #------------------------
    b=np.array([[ 0, 1, 0],
                [-1,-1, 1],
                [-1, 1, 1]])
    s = GameState(b,x=-1) # "O" player's turn
    n = MCNode(s) 
    for _ in range(100):
        e=n.sample(g)
        assert e==-1

    #------------------------
    b=np.array([[ 0, 1, 1],
                [ 0,-1, 1],
                [ 0,-1,-1]])

    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s) 
    v=0
    for _ in range(100):
        e=n.sample(g)
        assert e==-1 or e==1
        v+=e
    assert np.abs(v)<25  # X player has 1/2 chance to win and 1/2 to lose

    #------------------------
    # Terminal node, the game has already ended, the simulation result should always be the same.
    b=np.array([[-1, 0, 0],
                [ 1,-1, 1],
                [ 0, 1,-1]]) # terminal node: O player won
    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s) 
    for _ in range(100):
        assert n.sample(g)==-1
    
    b_=np.array([[-1, 0, 0],
                 [ 1,-1, 1],
                 [ 0, 1,-1]])
    assert np.allclose(n.s.b,b_) # the game state should not change after simulation


    b=np.array([[-1,-1, 1],
                [ 1, 1,-1],
                [-1, 1, 1]])
    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s)
    for _ in range(100):
        assert n.sample(g)==0

    #------------------------
    b=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=-1) # "O" player's turn
    n = MCNode(s) 
    v = 0
    for _ in range(1000):
        e = n.sample(g)
        assert e==-1 or e==1 or e==0
        v += e 
    assert np.abs(v-500) <100


    #-----------------------------
    # The AI agent should be compatible with both games: TicTacToe and Othello.
    # now let's test on the game "Othello":

    #---------------------
    # Game: Othello 
    g = Othello()  # game 

    #------------------------
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s)
    for _ in range(10):
        e=n.sample(g)
        assert e==-1

    #------------------------
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1) # "X" player's turn
    n = MCNode(s)
    w=0
    for _ in range(10):
        e=n.sample(g)
        w+=e
    assert np.abs(w)<9 # the two results (1 and -1) should roughly have the same chance



#-------------------------------------------------------------------------
def test_expand():
    '''(5 points) expand'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    # Current Node (root)
    b=np.array([[0, 1,-1],
                [0,-1, 1],
                [0, 1,-1]])
    s=GameState(b,x=1) #it's X player's turn
    n = MCNode(s) 
    # expand
    sc=n.expand(g)
    assert n.s.x==1
    assert len(n.c) ==3 

    assert type(sc)==MCNode
    assert sc.p == n
    assert sc.s.x==-1
    assert sc.p==n
    assert sc.c==[]
    assert sc.v==0
    assert sc.N==0

    b_=np.array([[0, 1,-1],
                 [0,-1, 1],
                 [0, 1,-1]])
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert c.s.x==-1
        assert c.p==n
        assert c.c==[]
        assert c.v==0
        assert c.N==0

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

    # the selected child node should be in the children list
    c = False
    for x in n.c:
        if sc==x:
            c=True
    assert c

    #--------------------------

    # Current Node (root)
    b=np.array([[ 1, 1,-1],
                [ 0,-1, 1],
                [ 0, 1,-1]])
    s=GameState(b,x=-1) #it's O player's turn
    n = MCNode(s)
    sc=n.expand(g)
    assert n.s.x==-1
    assert len(n.c) ==2
    assert type(sc)==MCNode
    assert sc.p == n
    assert sc.s.x==1
    assert sc.p==n
    assert sc.c==[]
    assert sc.v==0
    assert sc.N==0

    for c in n.c:
        assert c.s.x==1
        assert c.p==n
        assert c.c==[]
        assert c.v==0
        assert c.N==0

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

    # the selected child node should be in the children list
    c = False
    for x in n.c:
        if sc==x:
            c=True
    assert c

    #---------------------------
    s =GameState(np.zeros((3,3)),x=1)
    n = MCNode(s)
    sc=n.expand(g)
    assert n.s.x==1
    assert len(n.c) ==9
    a=False
    for c in n.c:
        assert c.s.x==-1
        assert c.p==n
        assert c.c==[]
        assert np.sum(c.s.b)==1
        assert c.v==0
        assert c.N==0
        if sc==c:
            a=True
    assert a # the selected child node should be in the children list



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
    b_=b.copy()
    s= GameState(b,x=1) #it's X player's turn
    n = MCNode(s) 
    # expand
    n.expand(g)
    assert len(n.c) ==2 
    assert n.s.x==1
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MCNode
        assert c.p==n
        assert c.c==[]
        assert c.v==0

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
    s = GameState(b,x=-1) #it's O player's turn
    n = MCNode(s) 
    # expand
    n.expand(g)
    print(n.c)
    assert len(n.c) ==3 
    assert n.s.x==-1
    # the current game state should not change after expanding
    assert np.allclose(n.s.b,b_) 
    for c in n.c:
        assert type(c)==MCNode
        assert c.p==n
        assert c.c==[]
        assert c.v==0

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




#-------------------------------------------------------------------------
def test_backprop():
    '''(5 points) backprop'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    b=np.array([[ 0, 0, 0],
                [-1, 1, 1],
                [ 0, 0,-1]])
    s = GameState(b,x=1) # X player's turn 
    r = MCNode(s) 
    r.expand(g) # expand the root node with one level of children nodes
    # simulation 1
    c1 = r.c[1] # suppose the second child node is selected 
    c1.backprop(e=1) # run a simulation on c, suppose the result is X player won
    assert c1.v ==1 # won one game in the simulation
    assert c1.N ==1 # number of simulations in the node 
    assert r.v ==1 
    assert r.N ==1


    # simulation 2
    c2 = r.c[2] # suppose the third child node is selected
    c2.expand(g) # expand the tree with one level of children nodes
    c2c0 = c2.c[0] # suppose the first grand child is selected
    c2c0.backprop(e=-1) # run a simulation, suppose the game result: O player won
    assert c2c0.v ==-1 
    assert c2c0.N ==1
    assert c2.v ==-1 
    assert c2.N ==1
    assert r.v ==0 
    assert r.N ==2
    assert c1.v ==1
    assert c1.N ==1
    c2c1 = c2.c[1] 
    assert c2c1.v ==0 
    assert c2c1.N ==0


    # simulation 3
    c2c1 = c2.c[1] # suppose the second child is selected
    c2c1.backprop(e=0) # run a simulation: a tie in the game
    assert c2c1.v ==0 
    assert c2c1.N ==1
    assert c2.v ==-1 
    assert c2.N ==2
    assert r.v ==0 
    assert r.N ==3
    assert c2c0.v ==-1 
    assert c2c0.N ==1
    assert c1.v ==1 # won one game in the simulation
    assert c1.N ==1 # number of simulations in the node 


#-------------------------------------------------------------------------
def test_compute_UCB():
    '''(5 points) compute_UCB'''

    # X player's turn, want to maximize the average score in vi/ni
    u = MCNode.compute_UCB(1,1,1,x=1)
    assert u==1.

    # O player's turn, want to minimize the average score in vi/ni
    u = MCNode.compute_UCB(-1,1,1,x=-1)
    assert u==1.

    # X player's turn, want to maximize the average score in vi/ni
    u = MCNode.compute_UCB(1,2,2,x=1)
    assert np.allclose(u,1.172, atol=1e-3)

    # O player's turn, want to minimize the average score in vi/ni
    u = MCNode.compute_UCB(-1,2,2,x=-1)
    assert np.allclose(u,1.172, atol=1e-3)

    # X player's turn, want to maximize the average score in vi/ni
    u = MCNode.compute_UCB(6,50,100,x=1)
    assert np.allclose(u,0.4665, atol=1e-3)

    # O player's turn, want to minimize the average score in vi/ni
    u = MCNode.compute_UCB(-6,50,100,x=-1)
    assert np.allclose(u,0.4665, atol=1e-3)

    # X player's turn, want to maximize the average score in vi/ni
    u = MCNode.compute_UCB(6,50,100,x=1,c=2.)
    assert np.allclose(u,0.7269, atol=1e-3)

    # O player's turn, want to minimize the average score in vi/ni
    u = MCNode.compute_UCB(-6,50,100,x=-1,c=2.)
    assert np.allclose(u,0.7269, atol=1e-3)

    u = MCNode.compute_UCB(0,0,100,x=1, c=2.)
    assert np.allclose(u,float('inf'), atol=1e-3)

    u = MCNode.compute_UCB(0,0,100,x=-1, c=2.)
    assert np.allclose(u,float('inf'), atol=1e-3)


#-------------------------------------------------------------------------
def test_select_a_child():
    '''(5 points) select_a_child'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    # A parent node with two children nodes 
    b=np.array([[ 1, 1,-1],
                [ 0,-1,-1],
                [ 0, 1, 1]])
    s = GameState(b,x=-1) # O player's turn
    p = MCNode(s) 
    p.expand(g) # expand the root node with one level of children nodes
    c1,c2=p.c

    # set the node statistics (this is only used for testing, in the real game, the statistics will be different from these numbers)
    p.v=-6
    p.N=12
    c1.v=-1
    c1.N=2
    c2.v=-5
    c2.N=10
    # select the child node with the highest UCB score
    c = p.select_a_child()
    assert c ==c1

    #----------------------
    p.v=-10
    p.N=20
    c1.v=-5
    c1.N=10
    c2.v=-5
    c2.N=10
    # select the child node with the highest UCB score
    c = p.select_a_child()
    assert c ==c1 # a tie in UCB score, use index as tie-breaker

    #----------------------
    p.v=-6
    p.N=20
    c1.v=-1
    c1.N=10
    c2.v=-5
    c2.N=10
    # select the child node with the highest UCB score
    c = p.select_a_child()
    assert c ==c2 

    #----------------------
    # A parent node with three children nodes 
    b=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [ 0,-1, 1]])
    s = GameState(b,x=1)  # X player's turn
    p = MCNode(s) 
    p.expand(g) # expand the root node with one level of children nodes
    c1,c2,c3=p.c

    p.v=1
    p.N=1
    c1.v=1
    c1.N=1
    c = p.select_a_child()
    assert c ==c2

    #----------------------
    p.v=2
    p.N=2
    c2.v=1
    c2.N=1
    c = p.select_a_child()
    assert c ==c3

    #----------------------
    p.v=1
    p.N=3
    c3.v=-1
    c3.N=1
    c = p.select_a_child()
    assert c ==c1

    #----------------------
    p.v=2
    p.N=4
    c1.v=2
    c1.N=2
    c = p.select_a_child()
    assert c == c2


#-------------------------------------------------------------------------
def test_selection():
    '''(5 points) selection'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    #----------------------
    # The root node is a leaf node
    b=np.array([[ 0,-1, 1],
                [ 0, 0, 1],
                [ 0,-1, 1]])
    s = GameState(b,x=-1) # O player's turn
    p = MCNode(s) 
    c = p.selection()
    assert c == p

    #----------------------
    # tree with one level of children nodes
    b=np.array([[ 1, 1,-1],
                [ 0,-1,-1],
                [ 0, 1, 1]])
    s= GameState(b,x=-1) # O player's turn
    p = MCNode(s) 
    p.expand(g) # expand the root node with one level of children nodes
    c1,c2=p.c

    p.v=-6
    p.N=12
    c1.v=-1
    c1.N=2
    c2.v=-5
    c2.N=10

    c = p.selection()
    assert c ==c1


    p.v=-10
    p.N=20
    c1.v=-5
    c1.N=10
    c2.v=-5
    c2.N=10
    # select the child node with the highest UCB score
    c = p.selection()
    assert c ==c1 # a tie in UCB score, use index as tie-breaker

    # A parent node with three children nodes 
    b=np.array([[ 0,-1,-1],
                [ 0, 1, 1],
                [ 0,-1, 1]])
    s = GameState(b,x=1) # X player's turn
    p = MCNode(s) 
    p.expand(g) # expand the root node with one level of children nodes
    c1,c2,c3=p.c

    p.v=1
    p.N=1
    c1.v=1
    c1.N=1
    c = p.selection()
    assert c ==c2

    #----------------------
    p.v=2
    p.N=2
    c2.v=1
    c2.N=1
    c = p.selection()
    assert c ==c3

    #----------------------
    p.v=1
    p.N=3
    c3.v=-1
    c3.N=1
    c = p.selection()
    assert c ==c1

    #----------------------
    p.v=2
    p.N=4
    c1.v=2
    c1.N=2
    c = p.selection()
    assert c == c2


    #----------------------
    # tree with two levels of children nodes
    b=np.array([[ 0, 0,-1],
                [-1, 1, 1],
                [ 0, 0, 0]])
    s = GameState(b,x=1) # X player's turn
    p = MCNode(s) 
    p.expand(g) # expand the root node with one level of children nodes
    p.v=0
    p.N=5
    for c in p.c:
        c.v=0
        c.N=1
        c.expand(g) # expand the second level children nodes

    for j in range(4): 
        for i in range(5): 
            l = p.selection()
            assert l==p.c[i].c[j]
            p.c[i].c[j].N=1
            p.c[i].N+=1
            p.N+=1

    p.c[1].v=1
    p.c[1].c[2].v=-1
    l = p.selection()
    assert l == p.c[1].c[2]
    

#-------------------------------------------------------------------------
def test_build_tree():
    '''(5 points) build_tree'''

    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    #--------------------------
    b=np.array([[ 0, 1, 1],
                [-1, 1,-1],
                [ 0,-1, 1]])
    s = GameState(b,x=-1) # O player's turn
    n = MCNode(s) 
    # run one iteration 
    n.build_tree(g,1)
    assert len(n.c)==2 
    assert n.N == 1
    assert n.v == 1

    for c in n.c:
        assert c.s.x==1
        assert c.p==n
        assert c.c==[]

    c = 0 
    for x in n.c:
        if x.N>0:
            c+=1
            assert x.v==1
            assert x.N==1
    assert c==1

    # run another iteration 
    n.build_tree(g,1)
    assert len(n.c)==2 
    assert n.N == 2
    assert n.v == 2

    for c in n.c:
        assert c.s.x==1
        assert c.p==n

    c = 0 
    for x in n.c:
        assert x.v==1
        assert x.N==1
        if len(x.c)==1:
            c+=1
    assert c==1

    # run two more iterations
    n.build_tree(g,2)
    assert n.N == 4
    assert n.v == 4

    for c in n.c:
        assert c.s.x==1
        assert c.p==n

    count=0
    for x in n.c:
        assert len(x.c)==1
        assert x.v==2
        assert x.N==2
        c=x.c[0]
        assert c.s.x==-1
        assert c.p==x
        assert c.N==c.v 
        assert c.N==1 or c.N==2
        if c.N==2:
            count+=1
    assert count ==1

    #--------------------------
    b=np.array([[ 0, 1, 1],
                [ 0,-1, 0],
                [ 0, 0, 0]])
    s= GameState(b,x=-1)
    n = MCNode(s)
    n.build_tree(g,1000)
    assert n.s.x ==-1
    assert n.N == 1000

    b1=np.array([[-1, 1, 1],
                 [ 0,-1, 0],
                 [ 0, 0, 0]])

    for x in n.c:
        if np.allclose(x.s.b,b1):
            assert x.s.x == 1
            assert x.p == n
            assert x.N > 800
            assert np.abs(x.v) < 50
            c1=x
   
    b2=np.array([[-1, 1, 1],
                 [ 0,-1, 0],
                 [ 0, 0, 1]]) 
    for x in c1.c:
        if np.allclose(x.s.b,b2):
            assert x.s.x ==-1
            assert x.p == c1
            assert x.N > 700
            assert np.abs(x.v) < 50
            c2=x

    b3=np.array([[-1, 1, 1],
                 [ 0,-1,-1],
                 [ 0, 0, 1]]) 
    for x in c2.c:
        if np.allclose(x.s.b,b3):
            assert x.s.x == 1
            assert x.p == c2
            assert x.N > 600
            assert np.abs(x.v) < 50
            c3=x

    b4=np.array([[-1, 1, 1],
                 [ 1,-1,-1],
                 [ 0, 0, 1]]) 
    for x in c3.c:
        if np.allclose(x.s.b,b4):
            assert x.s.x == -1
            assert x.p == c3
            assert x.N > 500
            assert np.abs(x.v) < 50


#-------------------------------------------------------------------------
def test_choose_optimal_move():
    '''(5 points) choose_optimal_move()'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 

    p=MCTSPlayer()
    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=1)
    n = MCNode(s)
    n.build_tree(g,100)
    r,c=p.choose_optimal_move(n)
    assert r == 2
    assert c == 0

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 1,-1],
                [ 0, 1,-1]])
    s = GameState(b,x=-1)
    n = MCNode(s)
    n.build_tree(g,100)
    r,c=p.choose_optimal_move(n)
    assert r == 2
    assert c == 0

    #-------------------------
    b=np.array([[ 1,-1, 1],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=-1)
    n = MCNode(s)
    n.build_tree(g,200)
    r,c=p.choose_optimal_move(n)
    assert r == 1
    assert c == 1

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
    n = MCNode(s) 
    n.build_tree(g,100)
    assert np.allclose(s.b,b_)
    r,c=p.choose_optimal_move(n)
    assert r == 0
    assert c == 0



#-------------------------------------------------------------------------
def test_MCTS_choose_a_move():
    '''(10 points) MCTS choose_a_move'''
    #---------------------
    # Game: TicTacToe
    g = TicTacToe()  # game 


    p =MCTSPlayer()
    b=np.array([[ 0,-1,-1],
                [ 0, 1, 0],
                [ 0, 0, 0]])
    s = GameState(b,x=1) 
    r,c=p.choose_a_move(g,s)
    assert r ==0
    assert c ==0

    b=np.array([[ 0, 0,-1],
                [ 0, 1,-1],
                [ 0, 0, 0]])
    s = GameState(b,x=1) 
    r,c=p.choose_a_move(g,s)
    assert r ==2
    assert c ==2


    b=np.array([[ 0, 0, 1],
                [ 0,-1, 1],
                [ 0, 0, 0]])
    s = GameState(b,x=-1) 
    r,c=p.choose_a_move(g,s)
    assert r ==2
    assert c ==2



    p1 = MCTSPlayer()
    p2 = RandomPlayer()
    p3 = MiniMaxPlayer()

    '''random vs MCTS'''
    for i in range(10):
        b=np.array([[ 0,-1, 1],
                    [-1, 1,-1],
                    [ 0,-1,-1]])
        s = GameState(b,x=1)
        e = g.run_a_game(p1,p2,s)
        assert e==1

    for i in range(10):
        b=np.array([[ 0,-1, 1],
                    [-1, 1,-1],
                    [-1, 1, 0]])
        s = GameState(b,x=1)
        e = g.run_a_game(p1,p2,s)
        assert e==0 

    ''' Minimax vs MCTS '''

    for i in range(10):
        b=np.array([[ 0, 0, 1],
                    [ 0,-1, 0],
                    [ 1,-1, 0]])
        s = GameState(b,x=1)
        e = g.run_a_game(p1,p3,s)
        assert e==0


    w=0
    for i in range(10):
        b=np.array([[ 0, 0, 0],
                    [ 0, 0, 0],
                    [ 1,-1, 0]])
        s = GameState(b,x=1)
        e = g.run_a_game(p1,p3,s)
        w += e
    assert w>1


    ''' MCTS vs MCTS '''
    w=0
    for i in range(10):
        b=np.array([[ 0, 0, 0],
                    [ 1,-1, 0],
                    [ 0, 0, 0]])
        s = GameState(b,x=1)
        e = g.run_a_game(p1,p1,s)
        w += e
    assert np.abs(w)<5

    ''' MCTS(n_iter=1) vs MCTS(n_iter=100) '''

    pm1 = MCTSPlayer(1)
    pm100 = MCTSPlayer(100)
    w=0
    for i in range(10):
        b=np.array([[ 0, 0, 0],
                    [ 0, 0, 0],
                    [ 1,-1, 0]])
        s = GameState(b,x=1)
        e = g.run_a_game(pm100,pm1,s)
        w += e
    assert np.abs(w)>4

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
        s = GameState(b.copy(),x=1)
        e = g.run_a_game(p1,p2,s)
        assert e==1

    ''' MCTS vs random'''
    s = GameState(b,x=1)
    e = g.run_a_game(p1,p2,s)
    assert e==1 



