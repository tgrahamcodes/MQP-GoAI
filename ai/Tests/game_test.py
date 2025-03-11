from ..game import *
import numpy as np
import sys
import time

#-------------------------------------------------------------------------
def test_python_version():
    assert sys.version_info[0]==3 # require python 3 (instead of python 2)

#-------------------------------------------------------------------------
def test_initial_game_state():
    '''initial_game_state()'''
    g=Othello()
    s = g.initial_game_state()
    n=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 1,-1, 0, 0, 0],
                [ 0, 0, 0,-1, 1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    assert np.allclose(s.b,n) 
    assert s.x==1


#-------------------------------------------------------------------------
def test_check_valid_move():
    '''check_valid_move()'''
    g=Othello()
    b=np.array([[ 1,-1, 0, 0, 0, 0,-1, 1],
                [-1, 0, 0,-1, 0, 0, 0,-1],
                [ 0, 0, 0,-1, 0,-1, 0, 0],
                [ 0, 0,-1,-1,-1, 0, 0, 0],
                [ 0,-1,-1, 1,-1,-1,-1, 0],
                [ 0, 0,-1,-1,-1,-1, 0, 0],
                [-1, 0, 0, 0, 0, 0, 0,-1],
                [ 1,-1, 0, 0, 1, 0,-1, 1]])
    s = GameState(b,x=1)

    assert g.check_valid_move(s,2,1) 
    assert g.check_valid_move(s,4,0) 
    assert g.check_valid_move(s,6,1) 
    assert g.check_valid_move(s,0,3) 
    assert g.check_valid_move(s,1,6) 
    assert g.check_valid_move(s,4,7) 
    assert g.check_valid_move(s,6,3) 
    assert g.check_valid_move(s,6,5) 

    assert g.check_valid_move(s,5,7) 
    assert g.check_valid_move(s,7,5) 
    assert g.check_valid_move(s,0,2) 
    assert g.check_valid_move(s,2,0) 
    assert g.check_valid_move(s,0,5) 
    assert g.check_valid_move(s,5,0) 
    assert g.check_valid_move(s,2,7) 
    assert g.check_valid_move(s,7,2) 

    assert not g.check_valid_move(s,0,0) 
    assert not g.check_valid_move(s,1,1) 

    s = GameState(b,x=-1)
    assert not g.check_valid_move(s,0,0) 
    assert not g.check_valid_move(s,6,3) 
    assert not g.check_valid_move(s,6,4) 
    assert not g.check_valid_move(s,6,5) 

    s = GameState(-b,x=-1)
    assert g.check_valid_move(s,2,1) 
    assert g.check_valid_move(s,4,0) 
    assert g.check_valid_move(s,6,1) 
    assert g.check_valid_move(s,0,3) 
    assert g.check_valid_move(s,1,6) 
    assert g.check_valid_move(s,4,7) 
    assert g.check_valid_move(s,6,3) 
    assert g.check_valid_move(s,6,5) 

    assert g.check_valid_move(s,5,7) 
    assert g.check_valid_move(s,7,5) 
    assert g.check_valid_move(s,0,2) 
    assert g.check_valid_move(s,2,0) 
    assert g.check_valid_move(s,0,5) 
    assert g.check_valid_move(s,5,0) 
    assert g.check_valid_move(s,2,7) 
    assert g.check_valid_move(s,7,2) 


    b=np.array([[ 1,-1,-1,-1,-1,-1,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    assert g.check_valid_move(s,0,7) 


#-------------------------------------------------------------------------
def test_get_valid_moves():
    '''get_valid_moves()'''
    g=Othello()
    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1,-1,-1, 0, 0],
                [ 0, 0, 0, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    s = GameState(b,x=1)# "X" player's turn
    m=g.get_valid_moves(s) 
    assert type(m)==list
    assert len(m)==5
    m.sort()
    assert m[0]== (2,2)
    assert m[1]== (2,3)
    assert m[2]== (2,4)
    assert m[3]== (2,5)
    assert m[4]== (2,6)


    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1, 1, 0, 0, 0],
                [ 0, 0, 0, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    s = GameState(b,x=-1)# "O" player's turn
    m=g.get_valid_moves(s) 
    assert type(m)==list
    assert len(m)==3
    m.sort()
    assert m[0]== (3,5)
    assert m[1]== (5,3)
    assert m[2]== (5,5)

    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1,-1,-1, 1, 0],
                [ 0, 0, 0, 1, 1, 1, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])

    s = GameState(b,x=1)# "X" player's turn
    m=g.get_valid_moves(s)
    assert type(m)==list
    assert len(m)==6
    m.sort()
    assert m[0]== (2,2)
    assert m[1]== (2,3)
    assert m[2]== (2,4)
    assert m[3]== (2,5)
    assert m[4]== (2,6)
    assert m[5]== (3,2)


#-------------------------------------------------------------------------
def test_check_game():
    '''check_game()'''

    g=Othello()
    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0, 0],
                [ 0, 0, 0, 1, 0, 0, 0, 0]])
    s = GameState(b,x=1)# "X" player's turn
    e=g.check_game(s) 
    assert e==1
    s = GameState(-b,x=1)# "X" player's turn
    e=g.check_game(s)
    assert e==-1
    s.b[3,3]=1
    e=g.check_game(s) 
    assert e==None

    s.b[0,3]=1
    s.b[1,3]=1
    e=g.check_game(s) 
    assert e==0




#-------------------------------------------------------------------------
def test_apply_a_move():
    '''apply_a_move()'''
    g=Othello()

    b=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 1, 0, 1, 0, 1, 0, 0],
                [ 0, 0,-1,-1,-1, 0, 0, 0],
                [ 0, 1,-1, 0,-1,-1,-1, 1],
                [ 0, 0,-1,-1,-1,-1, 1, 0],
                [ 0,-1, 0, 1, 0, 1, 0, 0],
                [ 1, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    g.apply_a_move(s,r=3,c=3) 
    n=np.array([[ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 1, 0, 1, 0, 1, 0, 0],
                [ 0, 0, 1, 1, 1, 0, 0, 0],
                [ 0, 1, 1, 1, 1, 1, 1, 1],
                [ 0, 0, 1, 1, 1,-1, 1, 0],
                [ 0, 1, 0, 1, 0, 1, 0, 0],
                [ 1, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(s.b,n)
    assert s.x==-1



#-------------------------------------------------------------------------
def test_dummy_choose_a_move():
    '''dummy_choose_a_move()'''
    g=Othello()
    b=np.array([[ 0,-1, 1,-1, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    p = DummyPlayer()
    s = GameState(b,x=1)
    r,c=p.choose_a_move(g,s)
    assert r==0
    assert c==4


#-------------------------------------------------------------------------
def test_get_move_state_pairs():
    '''get_move_state_pairs()'''
    g=Othello()
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    p = g.get_move_state_pairs(s)
    assert len(p)==2
    p1,p2 = p
    m1,s1 = p1
    m2,s2 = p2
    assert m1 == (0,1) 
    assert m2 == (0,5) 

    b1=np.array([[ 0, 1, 1, 1,-1, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(s1.b,b1)

    b2=np.array([[ 0, 0,-1, 1, 1, 1, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0],
                 [ 0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.allclose(s2.b,b2)


#-------------------------------------------------------------------------
def test_run_a_game():
    '''run_a_game()'''
    g=Othello()
    p = DummyPlayer()
    # test whether we can run a game using dummy player
    b=np.array([[ 0, 0,-1, 1,-1, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        assert e==-1



    b=np.array([[ 0, 0, 0, 0,-1, 1,-1, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0],
                [ 0, 0, 0, 0, 0, 0, 0, 0]])
    s = GameState(b,x=1)
    for i in range(10):
        e = g.run_a_game(p,p,s=s)
        assert e==1


#-------------------------------------------------------------------------
def test_go_initial_game_state():
    '''go_initial_game_state()'''
    g=GO(board_size=3)
    s = g.initial_game_state()
    n=np.array([[ 0, 0, 0],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    assert np.allclose(s.b,n) 
    assert s.p is None
    assert s.a==0 
    assert s.x==1 

    g=GO(board_size=2)
    s = g.initial_game_state()
    n=np.array([[ 0, 0],
                [ 0, 0]])
    assert np.allclose(s.b,n) 
    assert s.p is None
    assert s.a==0 
    assert s.x==1 

#-------------------------------------------------------------------------
def test_go_get_group():
    '''go_get_group()'''
    g=GO(board_size=3)
    s=np.array([[ 1, 0, 1],
                [ 0, 1, 0],
                [ 1, 0, 1]])
    l=g.get_group(s,1,1)
    assert len(l)==1 
    assert l.pop() == (1,1) 

    s=np.array([[ 1, 0, 0],
                [ 0, 1, 1],
                [ 1, 0, 0]])
    l=g.get_group(s,1,1)
    assert len(l)==2 
    assert (1,1) in l
    assert (1,2) in l

    s=np.array([[ 0, 1, 0],
                [ 0, 1, 1],
                [ 1, 0, 0]])
    l=g.get_group(s,1,1)
    assert len(l)==3 
    assert (1,1) in l
    assert (1,2) in l
    assert (0,1) in l

    s=np.array([[ 1, 1, 0],
                [ 0, 1, 1],
                [ 1, 0, 0]])
    l=g.get_group(s,1,1)
    assert len(l)==4 
    assert (1,1) in l
    assert (1,2) in l
    assert (0,1) in l
    assert (0,0) in l


    s=np.array([[ 1, 1, 0],
                [ 1, 1, 1],
                [ 1, 0, 0]])
    l=g.get_group(s,1,1)
    assert len(l)==6 
    assert (1,1) in l
    assert (1,2) in l
    assert (0,1) in l
    assert (0,0) in l
    assert (1,0) in l
    assert (2,0) in l

    l=g.get_group(-s,1,1)
    assert len(l)==6 
    assert (1,1) in l
    assert (1,2) in l
    assert (0,1) in l
    assert (0,0) in l
    assert (1,0) in l
    assert (2,0) in l



#-------------------------------------------------------------------------
def test_go_get_liberties():
    '''go_get_liberties()'''
    g=GO(board_size=3)
    s=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0, 0, 0]])
    gr = g.get_group(s,1,1)
    l=g.get_liberties(s,gr)
    assert l==4 

    s=np.array([[ 0, 0, 0],
                [ 0, 1, 0],
                [ 0,-1, 0]])
    gr = g.get_group(s,1,1)
    l=g.get_liberties(s,gr)
    print(l)
    assert l==3 

    s=np.array([[ 0, 0, 0],
                [ 0, 1,-1],
                [ 0,-1, 0]])
    gr = g.get_group(s,1,1)
    l=g.get_liberties(s,gr)
    assert l==2 

    s=np.array([[ 0, 0, 0],
                [-1, 1,-1],
                [ 0,-1, 0]])
    gr = g.get_group(s,1,1)
    l=g.get_liberties(s,gr)
    assert l==1 

    s=np.array([[ 0,-1, 0],
                [-1, 1,-1],
                [ 0,-1, 0]])
    gr = g.get_group(s,1,1)
    l=g.get_liberties(s,gr)
    assert l==0 

    s=np.array([[ 0, 0, 0],
                [ 0, 1,-1],
                [ 0, 0, 0]])
    gr = g.get_group(s,1,2)
    l=g.get_liberties(s,gr)
    assert l==2 

    s=np.array([[ 0, 0, 0],
                [ 0, 1,-1],
                [ 0, 0, 1]])
    gr = g.get_group(s,1,2)
    l=g.get_liberties(s,gr)
    assert l==1 



    s=np.array([[ 0, 0, 0],
                [ 0, 0, 0],
                [ 0, 0, 1]])
    gr = g.get_group(s,2,2)
    l=g.get_liberties(s,gr)
    assert l==2 



    s=np.array([[ 0, 0, 0],
                [ 0, 0, 1],
                [ 0, 0, 0]])
    gr = g.get_group(s,1,2)
    l=g.get_liberties(s,gr)
    assert l==3 

    s=np.array([[ 1, 0, 0],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    gr = g.get_group(s,0,0)
    l=g.get_liberties(s,gr)
    assert l==2 

    s=np.array([[ 0, 0, 0],
                [ 1, 0, 0],
                [ 0, 0, 0]])
    gr = g.get_group(s,1,0)
    l=g.get_liberties(s,gr)
    assert l==3 



#-------------------------------------------------------------------------
def test_go_remove_group():
    '''go_remove_group()'''
    g=GO(board_size=3)
    s=np.array([[ 1, 1, 0],
                [ 0, 1, 1],
                [ 1, 0, 1]])
    gr = g.get_group(s,1,1)
    g.remove_group(s,gr)
    n=np.array([[ 0, 0, 0],
                [ 0, 0, 0],
                [ 1, 0, 0]])
    assert np.allclose(s,n) 

#-------------------------------------------------------------------------
def test_go_check_valid_move():
    '''go_check_valid_move()'''
    g=GO(board_size=3)
    b=np.array([[ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0]])
    s = GO_state(b,x=1) 
    assert not g.check_valid_move(s,0,1)
    assert g.check_valid_move(s,1,1) 
    s = GO_state(b,x=-1) 
    assert not g.check_valid_move(s,1,1)

    g=GO(board_size=4)
    b=np.array([[ 0, 1,-1, 0],
                [ 1, 0, 1,-1],
                [ 0, 1,-1, 0],
                [ 0, 0, 0, 0]])
    s = GO_state(b,x=-1) 

    assert g.check_valid_move(s,1,1)

    b=np.array([[ 0, 1, 0, 0],
                [ 1, 0, 1, 0],
                [ 1,-1, 1, 0],
                [ 0, 1, 0, 0]])
    s = GO_state(b,x=-1) 
    assert not g.check_valid_move(s,1,1)

    b=np.array([[ 0, 1,-1, 0],
                [ 1, 0, 1,-1],
                [ 0, 1,-1, 0],
                [ 0, 0, 0, 0]])
    s = GO_state(b,x=-1,p=(1,1)) 
    assert not g.check_valid_move(s,1,1)

    # if player chooses to pass, r = None and c = None
    s = GO_state(b,x=-1) 
    assert g.check_valid_move(s,None,None)
    s = GO_state(b,x=1) 
    assert g.check_valid_move(s,None,None)

#-------------------------------------------------------------------------
def test_go_get_valid_moves():
    '''go_get_valid_moves()'''
    g=GO(board_size=3)
    b=np.array([[ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0]])

    s = GO_state(b,x=1) 
    m=g.get_valid_moves(s) # "X" player's turn
    assert type(m)==list
    assert len(m)==6
    assert m[0][0]==None
    assert m[0][1]==None
    m= m[1:]
    m.sort()
    assert m[0]== (0,0)
    assert m[1]== (0,2)
    assert m[2]== (1,1)
    assert m[3]== (2,0)
    assert m[4]== (2,2)

    s = GO_state(b,x=-1) 
    m=g.get_valid_moves(s) # "O" player's turn
    assert len(m)==1
    assert m[0][0]==None
    assert m[0][1]==None


    g=GO(board_size=4)
    b=np.array([[ 0, 1,-1, 0],
                [ 1, 0, 1,-1],
                [ 1, 1,-1, 0],
                [ 0,-1, 1, 1]])
    s = GO_state(b,x=-1) 
    m=g.get_valid_moves(s) # "O" player's turn
    assert len(m)==4
    assert m[0][0]==None
    assert m[0][1]==None
    m= m[1:]
    m.sort()
    assert m[0]== (0,3)
    assert m[1]== (1,1)
    assert m[2]== (2,3)


    g=GO(board_size=4)
    s = g.initial_game_state()
    m=g.get_valid_moves(s)
    assert len(m)==17



#-------------------------------------------------------------------------
def test_go_apply_a_move():
    '''go_apply_a_move()'''
    g=GO(board_size=3)
    b=np.array([[ 0, 0, 0],
                [ 1,-1, 1],
                [ 0, 1, 0]])
    s = GO_state(b,x=1) 
    g.apply_a_move(s,0,1)
    assert len(s.p)==2
    assert s.x ==-1
    n=np.array([[ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0]])
    assert np.allclose(s.b,n) 
    assert s.p == (1,1)


    g=GO(board_size=4)
    b=np.array([[ 0, 0, 0, 0],
                [ 1,-1, 1, 0],
                [ 1,-1, 1, 0],
                [ 0, 1, 0, 0]])
    s = GO_state(b,1) 
    g.apply_a_move(s,0,1)
    assert s.x ==-1

    n=np.array([[ 0, 1, 0, 0],
                [ 1, 0, 1, 0],
                [ 1, 0, 1, 0],
                [ 0, 1, 0, 0]])
    assert np.allclose(s.b,n) 

    b=np.array([[ 0, 0, 1, 0],
                [ 1,-1,-1, 1],
                [ 1,-1,-1, 1],
                [ 0, 1, 1, 0]])
    s = GO_state(b,1) 
    g.apply_a_move(s,0,1)
    n=np.array([[ 0, 1, 1, 0],
                [ 1, 0, 0, 1],
                [ 1, 0, 0, 1],
                [ 0, 1, 1, 0]])
    assert np.allclose(s.b,n) 

    # if player choose to pass
    s = GO_state(b,1) 
    g.apply_a_move(s,None,None)
    assert np.allclose(s.b,n) 
    assert s.x== -1
    assert s.a==1
    
    s.p=(2,3)
    g.apply_a_move(s,None,None)
    assert s.x == 1
    assert s.a== 2

    g=GO(board_size=4)
    b=np.array([[ 0, 1,-1, 0],
                [ 1,-1, 0,-1],
                [ 1, 1,-1, 0],
                [ 0,-1, 1, 1]])
    s = GO_state(b,x=1) 
    g.apply_a_move(s,1,2)
    assert s.p == (1,1)

    g.apply_a_move(s,0,0)
    assert s.p is None


#-------------------------------------------------------------------------
def test_go_is_surrounded():
    '''go_is_surrounded()'''
    g=GO(board_size=3)
    b=np.array([[ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0]])
    
    r=g.is_surrounded(b,{(1,1)})
    assert r==1
    r=g.is_surrounded(-b,{(1,1)})
    assert r==-1

    b=np.array([[ 0, 1, 0],
                [ 1, 0,-1],
                [ 0, 1, 0]])
    
    r=g.is_surrounded(b,{(1,1)})
    assert r==0

    b=np.array([[ 0, 0, 0],
                [ 0, 0, 0],
                [ 0, 0, 0]])
    l = g.get_group(b,1,1) 
    r=g.is_surrounded(b,l)
    assert r==0


    g=GO(board_size=4)
    b=np.array([[-1, 1, 0, 0],
                [ 1, 0, 1, 0],
                [ 1, 0, 1, 0],
                [ 0, 1, 0, 0]])
    l = g.get_group(b,1,1) 
    r=g.is_surrounded(b,l)
    assert r==1
    r=g.is_surrounded(-b,l)
    assert r==-1

    b=np.array([[ 0, 1, 0, 0],
                [ 1, 0, 1, 0],
                [ 1, 0, 0, 1],
                [ 0, 1,-1, 0]])
    l = g.get_group(b,1,1) 
    print(l)
    assert len(l) ==3
    r=g.is_surrounded(b,l)
    assert r==0

#-----------------------------------------------------------
def test_go_compute_score():
    '''go_compute_score()'''
    g=GO(board_size=3)
    b=np.array([[ 1, 1, 1],
                [ 1, 1, 1],
                [ 1, 1, 1]])
    t=g.compute_score(b,x=1)
    assert t==9

    b=np.array([[ 1, 1, 1],
                [ 1, 0, 1],
                [ 1, 1, 1]])
    t=g.compute_score(b,x=1)
    assert t==9


    b=np.array([[ 1, 1, 1],
                [ 1, 0,-1],
                [ 1, 1, 1]])
    t=g.compute_score(b,x=1)
    assert t==7
    t=g.compute_score(b,-1)
    assert t==1

    b=np.array([[ 0, 1, 0],
                [ 1, 0, 1],
                [ 0, 1, 0]])
    t=g.compute_score(b,x=1)
    assert t==9

    b=np.array([[ 0,-1, 0],
                [ 1, 0,-1],
                [ 0, 1, 0]])
    t=g.compute_score(b,x=1)
    assert t==3
    t=g.compute_score(b,-1)
    assert t==3


#-----------------------------------------------------------
def test_go_check_game():
    '''go_check_game()'''
    g=GO(board_size=3)
    b=np.array([[ 1, 1, 1],
                [ 1, 1, 1],
                [ 1, 1, 1]])
    s = GO_state(b,a=2) 
    e=g.check_game(s)
    assert e==1

    b=np.array([[ 0,-1, 0],
                [ 1, 0,-1],
                [ 1, 1, 0]])
    s = GO_state(b,a=2) 
    e=g.check_game(s)
    assert e==0

    b=np.array([[ 0,-1, 0],
                [ 1,-1,-1],
                [ 1, 1, 0]])
    s = GO_state(b,a=2) 
    e=g.check_game(s)
    assert e==-1

    s = GO_state(b,a=1) 
    e=g.check_game(s)
    assert e==None


#-----------------------------------------------------------
def test_go_speed():
    '''go_speed()'''
    start_time = time.time()
    p = DummyPlayer()
    n_games = 10 
    for _ in range(n_games):
        g=GO(max_game_length=150)
        g.run_a_game(p,p)
    end_time = time.time() - start_time 
    games_per_second = n_games/end_time
    print(f"Games per second: {games_per_second}")
    assert games_per_second > 0.1  # Ensure we can play at least 0.1 games per second

