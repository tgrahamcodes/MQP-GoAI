#-------------------------------------------------------------------------
import torch as th
import random

# ----------------------------------------------
def is_winning(b):
    return (th.sum(b,0)==3).any() or (th.sum(b,1)==3).any() or th.diagonal(b).sum()==3 or b[0,-1]+b[1,1]+b[-1,0]==3

#-------------------------------------------------------
class TicTacToe:
    '''
        TicTacToe game. The goal is to provide a platform for two AI players to play the game in turns and return the game result. 
        Here we assume the player will choose a move with the position ID at each time step.
        Here are the IDs of each location on the board of TicTacToe:
        ------------------
        |  0 |  1  |  2  |
        ------------------
        |  3 |  4  |  5  |
        ------------------
        |  6 |  7  |  8  |
        ------------------
        When X player chooses the location ID=7, the game will be like this:
        ------------------
        |    |     |     |
        ------------------
        |    |     |     |
        ------------------
        |    |  X  |     |
        ------------------
    '''
    # ----------------------------------------------
    def __init__(self):
        # game state (for x player) 
        s = th.zeros(3,3,3)
        s[-1] = 1
        self.s = s
        self.x = 1 # assuming x player is the first mover

    # ----------------------------------------------
    def check_game(self):
        '''
            check if the TicTacToe game has ended or not. 
            If yes (game ended), return the game result (1: x_player win, -1: o_player win, 0: draw)
            If no (game not ended yet), return None 
            Outputs:
                e: the result, an integer scalar with value 0, 1 or -1.
                    if e = None, the game has not ended yet.
                    if e = 0, the game ended with a draw.
                    if e = 1, X player won the game.
                    if e = -1, O player won the game.
        '''
        if is_winning(self.s[0]): # check x player
            return 1 # x player wins
        if is_winning(self.s[1]): # check O player
            return -1 # O player wins
        if self.s[2].sum()==0:
            return 0 # end with a draw/tie
        return None # game has not ended yet 

    # ----------------------------------------------
    def run_a_game(self,x_player,o_player):
        '''
            run a game. When the game ends, return the result of the game.
            Input:
                x_player: the "X" player (assuming to be the first mover in the game) 
                o_player: the "O" player (assuming to be the second mover in the game) 
            Outputs:
                r: the result of the game, an integer scalar with value 0, 1 or -1.
                    if r = 0, the game ends with a draw/tie.
                    if r = 1, X player wins the game.
                    if r = -1, O player wins the game.
        '''
        for _ in range(10):
            # check if the game has already ended
            r = self.check_game() # check if the game has ended already
            if r is not None: # if the game has ended, stop the game and return the result
                return r 
            # let the current player choose a move
            if self.x: # who's turn
                move_id = x_player.choose_a_move(self.s.clone()) # "X" player choose a move
            else:
                move_id = o_player.choose_a_move(self.s[[1,0,2]]) # "O" player choose a move
            assert list(move_id.size()) ==[] # test whether action returned by the player is a scalar
            r = th.div(move_id,3).long() # row of the move
            c = th.fmod(move_id,3).long() # column of the move
            assert self.s[2,r,c] # the move must be valid
            # apply the move on the game
            self.s[2,r,c]=0
            if self.x:
                self.s[0,r,c]=1
                self.x = 0
            else:
                self.s[1,r,c]=1
                self.x = 1
        assert False # the game should have ended and returned before


