#-------------------------------------------------------------------------
from abc import ABC, abstractmethod
import numpy as np
import copy
import sys
import os
#-------------------------------------------------------------------------



#-------------------------------------------------------
class GameState:
    '''
       This is the parent class of the game state of all board games. It defines the basic interface (APIs) that each game state class should provide. 
        A game state of a board game contains the following properties:
            b: the board setting, an integer matrix 
               b[i,j] = 0 denotes that the i-th row and j-th column is empty
               b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
               b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
            x: who's turn in the current step of the game, x=1 if it is "X" player's turn 
                        -1 if it is "O" player in the game. 
    '''
    def __init__(self, b, x=1):
        '''
            Initialize the game state with the input parameters.
        '''
        self.b = b
        self.x = x 

    def __hash__(self):
        return hash(str(self.b) + str(self.x))

    def __eq__(self, other):
        return (str(self.b) + str(self.x) == other)
 

#-------------------------------------------------------
class BoardGame(ABC):
    '''
       This is the parent class of all board games. It defines the basic interface (APIs) that each board game class should provide. 
    '''
    def __init__(self):
        self.M = 1000 # maximum game length

    # ----------------------------------------------
    @abstractmethod
    def convert_index(self, idx):
        '''
           Convert a BoardGame index to row and column for the given game.  
            Return:
                r, c: Row, column for the given index.
        '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def initial_game_state(self):
        '''
           Create an initial game state.  
            Return:
                s: the initial state of the game, which is an object of a subclass of GameState.
        '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def check_valid_move(self,s,r,c):
        '''
            check if a move is valid or not.
            Return True if valid, otherwise return False.
            Input:
                s: the current state of the game, which is an object of a subclass of GameState.
                r: the row number of the move
                c: the column number of the move
                x: the role of the player, 1 if you are the "X" player in the game
                        -1 if you are the "O" player in the game. 
            Outputs:
                valid: boolean scalar, True (if the move is valid), otherwise False 
        '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def get_valid_moves(self,s):
        '''
           Get a list of available (valid) next moves from a game state.  
            Input:
                s: the current state of the game, which is an object of a subclass of GameState.
            Outputs:
                m: a list of possible next moves, where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
            For example, for the following game state, 
                  s.b= [[ 1 , 0 ,-1 ],
                        [-1 , 1 , 0 ],
                        [ 1 , 0 ,-1 ]]
            the valid moves are the empty grid cells: 
                (r=0,c=1) --- the first row, second column 
                (r=1,c=2) --- the second row, the third column 
                (r=2,c=1) --- the third row , the second column
            So the list of valid moves is m = [(0,1),(1,2),(2,1)]
        '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def check_game(self,s):
        '''
            check if the game has ended or not. 
            If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
            If no (game not ended yet), return None 
            
            Input:
                s: the current state of the game
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game has not ended yet.
                    if e = 0, the game ended with a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        pass

    # ----------------------------------------------
    @abstractmethod
    def apply_a_move(self,s,r,c):
        '''
            Apply a move of a player to the game and change the game state accordingly. 
            Input:
                s: the current state of the game, 
                r: the row number of the move, an integer scalar.
                c: the column number of the move, an integer scalar.
            Result:
                s: the game state in the next step of the game, after applying the move 

            For example, suppose the current game state is:
                  s.b=[[ 0, 1, 1],
                       [ 0,-1,-1],
                       [ 1,-1, 1]]
            and it's "O" player's turn s.x=-1.
            If the "O" player chooses the move (r=1,c=0), then after applying the move on the board,
            the game state becomes:
                  s.b=[[ 0, 1, 1],
                       [-1,-1,-1],
                       [ 1,-1, 1]]
                and s.x = 1 (X player's turn in the next step)
        '''
        pass

    # ----------------------------------------------
    def get_move_state_pairs(self,s):
        '''
           Get a list of (move, state) pairs, such as [(m1,s1),(m2,s2),...],   where each pair (mi, si) represents one possible next move from the current game state s.
            Input:
                s: the current state of the game, which is an object of a subclass of GameState.
            Outputs:
                p: a list of all possible (next move, next game state) pairs, 
                    where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
            For example, for the following game state s in TicTacToe, 
                  s.b= [[ 1 , 0 ,-1 ],
                        [-1 , 1 , 0 ],
                        [ 1 , 0 ,-1 ]]
                  s.x = 1 
            the valid moves are the empty grid cells: 
                (r=0,c=1) --- the first row, second column 
                (r=1,c=2) --- the second row, the third column 
                (r=2,c=1) --- the third row , the second column
            So the list of valid move-state pairs are p = [ (m1,s1),(m2,s2),(m3,s3)]
                  Where m1 = (0,1)
                  s1.b= [[ 1 , 1 ,-1 ],
                         [-1 , 1 , 0 ],
                         [ 1 , 0 ,-1 ]]   and s1.x = -1
                  m2 = (1,2)
                  s2.b= [[ 1 , 0 ,-1 ],
                         [-1 , 1 , 1 ],
                         [ 1 , 0 ,-1 ]]   and s2.x = -1
                  m3 = (2,1)
                  s3.b= [[ 1 , 0 ,-1 ],
                         [-1 , 1 , 0 ],
                         [ 1 , 1 ,-1 ]]   and s3.x = -1
        '''
        # get the list of valid next moves from the current game state
        m=self.get_valid_moves(s)
        p = []
        for r,c in m:
            # for each next move (r,c):
            sc = copy.deepcopy(s)
            # get the new game state and who's turn next after the move
            self.apply_a_move(sc,r,c)
            p.append(((r,c),sc))
        return p

    # ----------------------------------------------
    def run_a_game(self,x_player,o_player,s=None):
        '''
            run a game starting from the game state (s), letting X and O players to play in turns, until the game ends.
            When the game ends, return the result of the game.
            Input:
                s: the current state of a game, 
                    If you want to start the game with initial state (empty board), s = None
                    If you want to start the game with a special game state, set the s accordingly.
                        s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                        s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                        s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                        s.x denotes who's turn in the current step of the game
                x_player: the "X" player 
                o_player: the "O" player 
            Outputs:
                e: the result of the game, an integer scalar with value 0, 1 or -1.
                    if e = 0, the game ends with a draw/tie.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        # if the initial game state is not given, use the empty board as the starting state
        if s is None:
            s=self.initial_game_state()
        else: # if s is assigned, start the game from the given game state s
            s=copy.deepcopy(s)

        # start the game: 
        for _ in range(self.M):
            e = self.check_game(s) # check if the game has ended already
            if e is not None: # if the game has ended, stop the game and return the result
                break
            if s.x==1:
                r,c = x_player.choose_a_move(self,s) # "X" player choose a move
            else:
                r,c = o_player.choose_a_move(self,s) # "O" player choose a move
            assert self.check_valid_move(s,r,c) # the move must be valid
            self.apply_a_move(s,r,c) # apply the move and update game state

        from Players.mcts import MCTSPlayer
        if isinstance(x_player, MCTSPlayer) and x_player.mem.file != 'testing':
            x_player.mem.export_mem()
        if isinstance(o_player, MCTSPlayer) and x_player.mem.file != 'testing':
            o_player.mem.export_mem()
            
        from Players.qfcnn import QFcnnPlayer
        if isinstance(x_player, QFcnnPlayer):
            x_player.model.save_model(x_player.file)
        if isinstance(o_player, QFcnnPlayer):
            o_player.model.save_model(o_player.file)

        from Players.policynn import PolicyNNPlayer
        if isinstance(x_player, PolicyNNPlayer):
            x_player.model.save_model(x_player.file)
        if isinstance(o_player, PolicyNNPlayer):
            o_player.model.save_model(o_player.file)

        from Players.valuenn import ValueNNPlayer
        if isinstance(x_player, ValueNNPlayer):
            x_player.model.save_model(x_player.file)
        if isinstance(o_player, ValueNNPlayer):
            o_player.model.save_model(o_player.file)
        
        return e

    # ----------------------------------------------
    def run_game_reinforcement(self,x_player,o_player,s=None):

        # if the initial game state is not given, use the empty board as the starting state
        if s is None:
            s=self.initial_game_state()
        else: # if s is assigned, start the game from the given game state s
            s=copy.deepcopy(s)

        moves = []
        # start the game: 
        for i in range(self.M):
            e = self.check_game(s) # check if the game has ended already
            if e is not None: # if the game has ended, stop the game and return the result
                break
            if s.x==1:
                r,c = x_player.choose_a_move(self,s) # "X" player choose a move
                moves.append([s, r, c])
            else:
                r,c = o_player.choose_a_move(self,s) # "O" player choose a move
            assert self.check_valid_move(s,r,c) # the move must be valid
            self.apply_a_move(s,r,c) # apply the move and update game state
        
        return e, moves


#-------------------------------------------------------
class TicTacToe(BoardGame):
    '''
        TicTacToe game environment: the goal is to provide a platform for two AI players to play the game in turns and return the game result. 
    '''
    def __init__(self):
        super(TicTacToe, self).__init__()
        self.N = 3
        self.output_size = self.N**2
        self.channels = 3

    # ----------------------------------------------
    def convert_index(self, idx):
        '''
           Convert a BoardGame index to row and column for the given game.  
            Return:
                r, c: Row, column for the given index.
        '''
        r = int(idx // self.N)
        c = int(idx % self.N)
        return r,c

    # ----------------------------------------------
    def initial_game_state(self):
        '''
           Create an initial game state.  
            Return:
                s: the initial state of the game, an integer matrix (TicTacToe: shape 3 by 3)
                    s[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
        '''
        b = np.zeros((3,3)) # start with an empty board
        s = GameState(b,x=1) # 'X' Player moves first
        return s 

    #-------------------------------------------------------
    def check_valid_move(self,s,r,c):
        '''
            check if a move is valid or not.
            Return True if valid, otherwise return False.
            Input:
                s: the current state of the game, which is an object of GameState class.
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                    s.x: who's turn in this step of the game, 1 if "X" player's turn; -1 if "O" player's turn 
                r: the row number of the move
                c: the column number of the move
            Outputs:
                valid: boolean scalar, True (if the move is valid), otherwise False 
        '''
        return s.b[r][c]==0 # if the cell is empty, it is a valid move 

    #-------------------------------------------------------
    ''' 
        Utility Functions: Let's first implement some utility functions for Tic-Tac-Toe game. 
        We will need to use them later.
    '''
    # ----------------------------------------------
    def get_valid_moves(self, s):
        '''
           Get a list of available (valid) next moves from a game state of TicTacToe 
            Input:
                s: the current state of the game, which is an object of GameState class.
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                    For example, the following game state 
                     | X |   | O |
                     | O | X |   |
                     | X |   | O |
                    is represented as the following numpy matrix in game state
                    s.b= [[ 1 , 0 ,-1 ],
                          [-1 , 1 , 0 ],
                          [ 1 , 0 ,-1 ]]
            Outputs:
                m: a list of possible next moves, where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
            For example, for the following game state, 
                  s.b= [[ 1 , 0 ,-1 ],
                        [-1 , 1 , 0 ],
                        [ 1 , 0 ,-1 ]]
            the valid moves are the empty grid cells: 
                (r=0,c=1) --- the first row, second column 
                (r=1,c=2) --- the second row, the third column 
                (r=2,c=1) --- the third row , the second column
            So the list of valid moves is m = [(0,1),(1,2),(2,1)]
            Hint: you could use np.where() function to find the indices of the elements in an array, where a test condition is true.
            Hint: you could solve this problem using 2 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        rs,cs=np.where(s.b==0)
        m = list(zip(rs,cs))
        #########################################
        return m
    
    
        ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_get_valid_moves' in the terminal.  '''


    # ----------------------------------------------
    def check_game(self,s):
        '''
            check if the TicTacToe game has ended or not. 
            If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
            If no (game not ended yet), return None 
            
            Input:
                s: the current state of the game, which is an object of GameState class.
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                    s.x: who's turn in this step of the game, 1 if "X" player's turn; -1 if "O" player's turn 
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game has not ended yet.
                    if e = 0, the game ended with a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
            Hint: you could solve this problem using 11 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        b = s.b
        x = b.sum(axis=0)
        y = b.sum(axis=1)
        # check the 8 lines in the board to see if the game has ended.
        z = [ b[0,0] + b[1,1] + b[2,2], b[0,2] + b[1,1] + b[2,0]] 
        t=np.concatenate((x,y,z))
        # if the game has ended, return the game result 
        if np.any(t==3):
            return 1
        if np.any(t==-3):
            return -1
        if np.sum(b==0)==0:
            return 0
        # if the game has not ended, return None
        e = None 
        #########################################
        return e
    
        ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_check_game' in the terminal.  '''


    # ----------------------------------------------
    def apply_a_move(self,s,r,c):
        '''
            Apply a move of a player to the TicTacToe game and change the game state accordingly. 
            Input:
                s: the current state of the game, which is an object of GameState class.
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                    s.x: who's turn in this step of the game, 1 if "X" player's turn; -1 if "O" player's turn 
                r: the row number of the move, an integer scalar.
                c: the column number of the move, an integer scalar.
            Result:
                s: the game state in the next step of the game, after applying the move 

            For example, suppose the current game state is:
                  s.b=[[ 0, 1, 1],
                       [ 0,-1,-1],
                       [ 1,-1, 1]]
            and it's "O" player's turn s.x=-1.
            If the "O" player chooses the move (r=1,c=0), then after applying the move on the board,
            the game state becomes:
                  s.b=[[ 0, 1, 1],
                       [-1,-1,-1],
                       [ 1,-1, 1]]
                and s.x = 1 (X player's turn in the next step)
        '''
        assert self.check_valid_move(s,r,c) # check whether the step is valid or not
        s.b[r,c]=s.x # fill the empty cell with the current player's stone
        s.x *= -1 # two players take turns to play


#-------------------------------------------------------------------------
'''
    Othello is a larger board game than TicTacToe: https://en.wikipedia.org/wiki/Reversi 
    For a demo of the Othello game, you could visit: https://hewgill.com/othello/ 
    or https://www.mathsisfun.com/games/reversi.html  (for playing with an AI)
'''

#-------------------------------------------------------
class Othello(BoardGame):
    '''
        Othello game engine: the goal is to provide a platform for two AI players to play the game in turns and return the game result. 
    '''
    def __init__(self):
        super(Othello, self).__init__()
        self.N = 8
        self.output_size = self.N**2
        self.channels = 3

    # ----------------------------------------------
    def convert_index(self, idx):
        '''
           Convert a BoardGame index to row and column for the given game.  
            Return:
                r, c: Row, column for the given index.
        '''
        r = int(idx // self.N)
        c = int(idx % self.N)
        return r,c

    # ----------------------------------------------
    def initial_game_state(self):
        '''
           Create an initial game state.  
            Return:
                s: the initial state of the game, 
                   s.b is an integer matrix (8 by 8), showing the board setting
                   s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                   s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                   s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                   s.x denotes who's turn in the current step of the game
        '''
        b = np.zeros((8,8))
        b[3,3]= 1
        b[3,4]=-1
        b[4,3]=-1
        b[4,4]= 1
        s = GameState(b,x=1) # X player move first
        return s

    # ----------------------------------------------
    def check_valid_move(self,s,r,c):
        '''
            check if a move is valid or not.
            Return True if valid, otherwise return False.
            Input:
                s: the current state of the game,
                   s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                   s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                   s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                   s.x denotes who's turn in the current step of the game
                r: the row number of the move
                c: the column number of the move
            Outputs:
                valid: boolean scalar, True (if the move is valid), otherwise False 
        '''
        # the cell must be empty
        if s.b[r,c]!=0:
            return False
        # a move is valid only if it can flip at least one stone in any of the 8 directions
        # 8 directions to check if there is a possible flip
        direction =[( 1, 0),(-1, 0),
                    ( 0, 1),( 0,-1),
                    ( 1, 1),(-1,-1),
                    ( 1,-1),(-1, 1)]
        nx=-s.x
        # check 8 directions
        for ri,ci in direction:
            found_nx=False
            # check one direction
            for i in range(1,8):
                a=r+ri*i
                b=c+ci*i
                if a<0 or a>7 or b<0 or b>7: # out of the board
                    break
                if found_nx:
                    if s.b[a][b]==s.x: 
                        return True # valid (found a flip)
                    if s.b[a][b]==0:
                        break
                elif s.b[a][b]==nx: # found one
                        found_nx=True
                else:
                    break
        return False


    # ----------------------------------------------
    def get_valid_moves(self,s):
        '''
           Get a list of available (valid) next moves from a game state of Othello 
         
            Input:
                s: the current state of the game, 
                   s.b is an integer matrix (8 by 8), showing the board setting
                   s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                   s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                   s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                   s.x denotes who's turn in the current step of the game
            Outputs:
                m: a list of possible next moves, where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
            For example, suppose we have the following game state and it is "X" player's turn (s.x=1):
                  s.b= [[ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0,-1,-1,-1, 1, 0],
                        [ 0, 0, 0, 1, 1, 1, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0]]
            The valid moves are the empty grid cells, where at least one "O" player's stone can be flipped. 
            They are marked as "*": 
                       [[ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, *, *, *, *, *, 0],
                        [ 0, 0, *,-1,-1,-1, 1, 0],
                        [ 0, 0, 0, 1, 1, 1, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0],
                        [ 0, 0, 0, 0, 0, 0, 0, 0]]
            In the example, the valid next moves are
                (r=2,c=2) --- the third row, third column 
                (r=2,c=3) --- the third row, fourth column 
                (r=2,c=4) --- the third row, fifth column
                (r=2,c=5) --- the third row, sixth column
                (r=2,c=6) --- the third row, seventh column
                (r=3,c=2) --- the fourth row, third column
            So the list of valid moves is m = [(2,2),(2,3),(2,4),(2,5),(2,6),(3,2)]
        '''
        rs,cs=np.where(s.b==0) 
        e = list(zip(rs,cs)) #empty slots
        m = []
        for r,c in e:
            if self.check_valid_move(s,r,c):
                m.append((r,c))
        return m


    # ----------------------------------------------
    def check_game(self,s):
        '''
            check if the game has ended or not. 
            If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
            If no (game not ended yet), return None 
            
            Input:
                s: the current state of the game,
                   s.b is an integer matrix (8 by 8), showing the board setting
                   s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                   s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                   s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                   s.x denotes who's turn in the current step of the game
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game has not ended yet.
                    if e = 0, the game ended with a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        # if neither player has a valid move, the game ends
        s_x = GameState(s.b,x=1)
        s_o = GameState(s.b,x=-1)
        if len(self.get_valid_moves(s_x))==0 and len(self.get_valid_moves(s_o))==0:
            nx = np.sum(s.b==1)
            no = np.sum(s.b==-1)
            # the player with the most stones wins
            if nx>no:
                e=1
            elif nx<no:
                e=-1
            else:
                e=0
        else:
            e=None
        return e

    # ----------------------------------------------
    def apply_a_move(self,s,r,c):
        '''
            Apply a move of a player to the Othello game and change the game state accordingly. 
            Here we assume the move is valid.
            
            Input:
                s: the current state of the game,
                   s.b is an integer matrix (8 by 8), showing the board setting
                   s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                   s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                   s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                   s.x denotes who's turn in the current step of the game
                r: the row number of the move
                c: the column number of the move
        '''
        s.b[r,c]=s.x 
        # 8 directions to check
        direction =[( 1, 0),(-1, 0),
                    ( 0, 1),( 0,-1),
                    ( 1, 1),(-1,-1),
                    ( 1,-1),(-1, 1)]
        nx=-s.x
        f = []
        # flip 8 directions
        for ri,ci in direction:
            found_nx=False
            l=[]
            # flip one direction
            for i in range(1,8):
                a=r+ri*i
                b=c+ci*i
                if a<0 or a>7 or b<0 or b>7: # out of the board
                    break
                if found_nx:
                    if s.b[a][b]==s.x: 
                        for li in l:
                            f.append(li)
                        break 
                    elif s.b[a][b]==0:
                        break
                    else:
                        l.append((a,b))
                elif s.b[a][b]==nx: # found one
                        found_nx=True
                        l.append((a,b))
                else:
                    break
        for ri,ci in f:
            s.b[ri,ci]=s.x
        
        s.x*=-1
        # determine who's turn in the next step of the game
        if len(self.get_valid_moves(s))==0:
            s.x*=-1


#-------------------------------------------------------
class Player(ABC):
    '''
       This is the parent class of all board game players. It defines the basic interface (APIs) that each player class should provide. 
    '''

    # ----------------------------------------------
    @abstractmethod
    def choose_a_move(self,g,s):
        '''
            The action function, which chooses one random valid move in each step of the game.  
            This function will be called by the game at each game step.
            For example, suppose we have 2 random players (say A and B) in a game.
            The game will call the choose_a_move() function of the two players in turns as follows:

            Repeat until game ends:
                (1) r,c = A.choose_a_move(game,game_state, x=1 ) --- "X" player (A) choose a move
                (2) the game updates its game state 
                (3) r,c = B.choose_a_move(game,game_state, x=-1 ) --- "O" player (B) choose a move
                (4) the game updates its game state 

            Input:
                g: the game environment being played, such as TicTacToe or Othello. 
                s: the current state of the game, an integer matrix of shape 3 by 3 (TicTacToe) or 8 by 8 (Othello). 
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    For example, in TicTacToe, the following game state 
                     | X |   | O |
                     | O | X |   |
                     | X |   | O |
                    is represented as the following numpy matrix
                    s.b= [[ 1 , 0 ,-1 ],
                          [-1 , 1 , 0 ],
                          [ 1 , 0 ,-1 ]]
                    s.x: the role of the player, x=1 if this agent is the "X" player in the game
                         s.x=-1 if this agent is the "O" player in the game. 
           Outputs:
                r: the row number of the next move, an integer scalar.
                c: the column number of the next move, an integer scalar.
        '''
        pass



#-------------------------------------------------------
class DummyPlayer(Player):
    '''
        Dummy player: it always chooses the first valid move.
        This player is used for testing game engines. 
    '''
    # ----------------------------------------------
    def choose_a_move(self,g,s):
        '''
            The action function, which chooses one random valid move in each step of the game.  
            This function will be called by the game at each game step.
            For example, suppose we have 2 random players (say A and B) in a game.
            The game will call the choose_a_move() function of the two players in turns as follows:

            Repeat until game ends:
                (1) r,c = A.choose_a_move(game,game_state, x=1 ) --- "X" player (A) choose a move
                (2) the game updates its game state 
                (3) r,c = B.choose_a_move(game,game_state, x=-1 ) --- "O" player (B) choose a move
                (4) the game updates its game state 

            Input:
                g: the game environment being played, such as TicTacToe or Othello. 
                s: the current state of the game,
                    s.b is an integer matrix of shape 3 by 3 (TicTacToe) or 8 by 8 (Othello). 
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    For example, in TicTacToe, the following game state 
                     | X |   | O |
                     | O | X |   |
                     | X |   | O |
                    is represented as the following numpy matrix
                    s.b= [[ 1 , 0 ,-1 ],
                          [-1 , 1 , 0 ],
                          [ 1 , 0 ,-1 ]]
                   s.x: the role of the player, x=1 if this agent is the "X" player in the game
                        s.x=-1 if this agent is the "O" player in the game. 
           Outputs:
                r: the row number of the next move, an integer scalar.
                c: the column number of the next move, an integer scalar.
        '''
        m=g.get_valid_moves(s)
        r,c = m[-1]
        return r,c



#-------------------------------------------------------------------------
'''
    GO is a famous board game, very challenging problem in AI. 
    For a demo of the Othello game, you could visit: https://online-go.com  
'''
#-------------------------------------------------------
class GO_state(GameState):
    '''
        A game state of GO game 
            b: the board setting, an integer matrix 
               b[i,j] = 0 denotes that the i-th row and j-th column is empty
               b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
               b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
            x: who's turn in this step of the game, x = 1 if 'x' player's turn; x = -1 if 'O' player's turn
            p: the banned position to avoid repeated board setting 
            a: the number of passes by the players in the previous two steps of the game.
               Each time when a player chooses to pass without placing a stone on board, 'a' will increase a count by 1. 
               When 'a' = 2, both players passes in the previous two steps of the game, the game ends.
            t: the length of the current game, t starts with 0, and increase by one each step of the game
    '''
    def __init__(self, b, x=1, p=None, a=0,t=0):
        '''
            Initialize the game state with the input parameters.
        '''
        super(GO_state, self).__init__(b,x)
        self.p = p
        self.a = a 
        self.t = t 

    def __hash__(self):
        return hash(str(self.b) + str(self.x) + str(self.p) + str(self.a) + str(self.t))

    def __eq__(self, other):
        return (str(self.b) + str(self.x) + str(self.p) + str(self.a) + str(self.t) == other)

#-------------------------------------------------------
class GO(BoardGame):
    '''
        GO game engine: the goal is to provide a platform for two AI players to play the game in turns and return the game result. 
    '''
    def __init__(self, board_size=19,max_game_length=None):
        self.N = board_size 
        if max_game_length is None: 
            self.M = board_size*board_size*3 # maximum game length
        else:
            self.M = max_game_length
        self.output_size = self.N**2
        self.channels = 4

    # ----------------------------------------------
    def convert_index(self, idx):
        '''
           Convert a BoardGame index to row and column for the given game.  
            Return:
                r, c: Row, column for the given index.
        '''
        r = int(idx // self.N)
        c = int(idx % self.N)
        return r,c

    # ----------------------------------------------
    def initial_game_state(self):
        '''
           Create an initial game state.  
            Return:
                s: the initial state of the game, a GO_state object 
                    s.b: the board setting, an integer matrix 
                       b[i,j] = 0 denotes that the i-th row and j-th column is empty
                       b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                       b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.p: the banned position for next move to avoid repeated board settings (a game rule)
                        p = None, when there is no banned position 
                        p =  (r,c), when the position (r,c) is banned for next move 
                    s.a: the number of passes by the players in the previous two steps of the game 
        '''
        b = np.zeros((self.N,self.N), dtype=int)
        s = GO_state(b)
        return s

    # ----------------------------------------------
    def is_on_board(self,r,c):
        '''
           check whether a position is on board 
           Inputs:
                r: the row number of the move
                c: the column number of the move
            Return:
                b: boolean, true if the position is on board 
        '''
        return r>=0 and r<self.N and c>=0 and c<self.N 
    # ----------------------------------------------
    @staticmethod
    def neighbors(r,c):
        '''
           get the list of direct neighbors of a position on board 
           Inputs:
                r: the row number of the move
                c: the column number of the move
        '''
        return [(r,c+1), (r+1,c), (r,c-1),(r-1,c)]
    # ----------------------------------------------
    def get_group(self,b,r,c,x=None):
        '''
           Get the group of stones connected to the current stone
           Inputs:
                b: the state of the game, an integer matrix 
                    b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                r: the row number of the move
                c: the column number of the move
                x: the role of the player, 1 if you are the "X" player in the game
                        -1 if you are the "O" player in the game. 
                        None: if the role can be determined automatically by looking at the state of the game 
            Return:
                g: the group of stones connected to the current stone, a python set of {(r1,c1),(r2,c2),... }
        '''
        if x is None:
            x=b[r,c]
        g=set()
        def add_neighbor(g,b,r,c,x):
            if (r,c) in g:
                return
            g.add((r,c)) 
            # Breath-First-Search
            for nr,nc in self.neighbors(r,c): 
                if self.is_on_board(nr,nc) and b[nr,nc]==x:
                    # recursion
                    add_neighbor(g,b,nr,nc,x)
        add_neighbor(g,b,r,c,x)
        return g

    # ----------------------------------------------
    def get_liberties(self,b,g):
        '''
           Calculate the liberties of a group of stones. 
           Inputs:
                b: the state of the game, an integer matrix 
                    b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                g: the group of stones connected to the current stone, a python set of {(r1,c1),(r2,c2),... }
          Returns: 
                l: the number a liberties, an integer scaler.
        '''
        l = set()
        for r,c in g:
            for nr,nc in self.neighbors(r,c): 
                if self.is_on_board(nr,nc) and b[nr,nc]==0 and ( (nr,nc) not in g ) and ( (nr,nc) not in l ):
                    l.add((nr,nc))
        return len(l)

    # ----------------------------------------------
    def remove_group(self,b,g):
        '''
           remove the group of stones connected to the current stone
           Inputs:
                b: the state of the game, an integer matrix 
                    b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                g: the group of stones connected to the current stone, a python set of {(r1,c1),(r2,c2),... }
        '''
        for r,c in g:
            b[r,c]=0

    # ----------------------------------------------
    def check_valid_move(self,s,r,c):
        '''
            check if a move is valid or not.
            Return True if valid, otherwise return False.
            Input:
                s: the current state of the game, a GO_state object 
                    s.b: the board setting, an integer matrix 
                       b[i,j] = 0 denotes that the i-th row and j-th column is empty
                       b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                       b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.p: the banned position for next move to avoid repeated board settings (a game rule)
                        p = None, when there is no banned position 
                        p =  (r,c), when the position (r,c) is banned for next move 
                    s.a: the number of passes by the players in the previous two steps of the game 
                r: the row number of the move
                c: the column number of the move
                x: the role of the player, 1 if you are the "X" player in the game
                        -1 if you are the "O" player in the game. 
            Outputs:
                valid: boolean scalar, True (if the move is valid), otherwise False 
        '''
        x = s.x
        # if player chooses to pass in this step (r = None and c = None), it is a valid move
        if r is None or c is None:
            return True

        # if not empty (not allowed)
        if s.b[r,c]!=0:
            return False

        # if banned (not allowed) 
        if (s.p is not None) and r == s.p[0] and c == s.p[1]:
            return False

        # if suicide move without kill (not allowed)
        g = self.get_group(s.b,r,c,x)
        l = self.get_liberties(s.b,g)
        if l > 0:
            return True #  
        # if suicide move with kill (allowed) 
        for nr,nc in self.neighbors(r,c):
            if self.is_on_board(nr,nc) and s.b[nr,nc]==-x:
                g = self.get_group(s.b,nr,nc)
                l = self.get_liberties(s.b,g) 
                if l==1: # can kill
                    return True
        return False

    # ----------------------------------------------
    def get_valid_moves(self, s):
        '''
           Get a list of available (valid) next moves from a game state.  
            Input:
                s: the current state of the game, a GO_state object 
                    s.b: the board setting, an integer matrix 
                       b[i,j] = 0 denotes that the i-th row and j-th column is empty
                       b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                       b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.p: the banned position for next move to avoid repeated board settings (a game rule)
                        p = None, when there is no banned position 
                        p =  (r,c), when the position (r,c) is banned for next move 
                    s.a: the number of passes by the players in the previous two steps of the game 
               x: who's turn in this step of the game, x=1 if it is "X" player's turn. 
                    x=-1 if it's "O" player's turn. 
            Outputs:
                m: a list of possible next moves, where each next move is a (r,c) tuple, 
                   r denotes the row number, c denotes the column number. 
        '''
        # "pass" is a valid move ( player chooses not to place any stone)
        m = [(None,None)]
        rs,cs=np.where(s.b==0) 
        e = list(zip(rs,cs)) #empty slots
        for r,c in e:
            if self.check_valid_move(s,r,c):
                m.append((r,c))
        return m

    # ----------------------------------------------
    def is_surrounded(self,b,g):
        '''
           check whether a group is fully surrounded by a player 
           Inputs:
                b: the state of the game, an integer matrix 
                    b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                g: the group of stones connected to the current stone, a python set of {(r1,c1),(r2,c2),... }
          Returns: 
                r: the result, 1 if all surrounded by x player;  -1 if all surrounded by O player; 0 if by both or none 
        '''
        find_x = True # whether or not need to search for x player's stone
        find_o = True# whether or not need to search for o player's stone
        l = set()
        for r,c in g:
            for nr,nc in self.neighbors(r,c): 
                if not (find_x or find_o):
                    return 0
                if self.is_on_board(nr,nc) and ((find_x and b[nr,nc]==1) or (find_o and b[nr,nc]==-1)) and ( (nr,nc) not in g ) and ( (nr,nc) not in l ):
                    l.add((nr,nc))
                    if  b[nr,nc] == 1:
                        find_x = False 
                    else:
                        find_o = False 
        if find_x and find_o:
            return 0
        if find_x:
            return -1
        if find_o:
            return  1

    # ----------------------------------------------
    def compute_score(self,b,x):
        '''
           Calculate the score of the board setting. 
           Inputs:
                b: the state of the game, an integer matrix 
                    b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                x: which player to count for score, x =1, when computing score for 'X' player;
                    x = -1 when computing score for 'O' player
          Returns: 
                t: the territory of 'X' player, an integer scalar.
                    This number is sum of the number of crossings that 'X' player stone has taken and the empty crossings that are fully surrounded by 'x' player's stones. 
        '''
        # number of stones for the player
        t = np.sum(b==x)
        # check each empty crossing and see if it was surrounded by 'x' player's stones.
        rs,cs=np.where(b==0) 
        l = set()
        e = list(zip(rs,cs)) #empty slots
        for r,c in e:
            if (r,c) in l:
                continue
            # check if surrounded by 'x' player's stones only
            # find all the connected empty positions
            g = self.get_group(b,r,c)
            for z in g:
                l.add(z)
            # check if all surroundings are either wall or 'x' player's stones
            v = self.is_surrounded(b,g) 
            if v == x:
                t+=1
        return t

    # ----------------------------------------------
    def check_game(self,s):
        '''
            check if the game has ended or not. 
            If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
            If no (game not ended yet), return None 
            
            Input:
                s: the current state of the game, a GO_state object 
                    s.b: the board setting, an integer matrix 
                       b[i,j] = 0 denotes that the i-th row and j-th column is empty
                       b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                       b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.p: the banned position for next move to avoid repeated board settings (a game rule)
                        p = None, when there is no banned position 
                        p =  (r,c), when the position (r,c) is banned for next move 
                    s.a: the number of passes by the players in the previous two steps of the game 
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game has not ended yet.
                    if e = 0, the game ended with a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        # if both players choose to pass or reached maximum game length, game ends
        if s.a==2 or s.t==self.M:
            # count score
            s_x = self.compute_score(s.b,x=1)
            s_o = self.compute_score(s.b,x=-1)
            # in full sized board: x player need to lead by 3.75 in order to win
            if self.N ==19:
                s_o+=3.75
            if s_x > s_o:
                return 1 # X player wins
            elif s_x<s_o:
                return -1 # O player wins
            else:
                return 0 # draw
        else:
            return None

    # ----------------------------------------------
    def apply_a_move(self,s,r,c):
        '''
            Apply a move of a player to the game and change the game state accordingly. 
            Input:
                s: the current state of the game, a GO_state object 
                    s.b: the board setting, an integer matrix 
                       b[i,j] = 0 denotes that the i-th row and j-th column is empty
                       b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                       b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.p: the banned position for next move to avoid repeated board settings (a game rule)
                        p = None, when there is no banned position 
                        p =  (r,c), when the position (r,c) is banned for next move 
                    s.c: the number of passes by the players in the previous two steps of the game 
                r: the row number of the move, an integer scalar.
                c: the column number of the move, an integer scalar.
                x: who's turn in this step of the game, x=1 if it is "X" player's turn. 
                    x=-1 if it's "O" player's turn. 
        '''
        s.p=None
        x = s.x
        # other player's turn in the next step
        # if player choose to pass
        if r is None or c is None:
            s.a += 1
        else:
            # otherwise apply the move on the board
            s.a=0
            s.b[r,c]=x
            # check if any group get surrounded (killed)
            for nr,nc in self.neighbors(r,c):
                # for each neighboring point
                if self.is_on_board(nr,nc) and s.b[nr,nc]==-x:
                    # compute the liberties of the group that point becomes to
                    g = self.get_group(s.b,nr,nc)
                    l = self.get_liberties(s.b,g)
                    # if the liberties of a group becomes 0
                    if l == 0:
                        # kill the group
                        self.remove_group(s.b,g)
                        if len(g)==1:
                            s.p=g.pop()
        s.x*=-1
        s.t+=1 # game step +1

