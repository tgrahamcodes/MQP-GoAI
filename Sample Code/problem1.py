
#-------------------------------------------------------------------------
import numpy as np
from game import GameState,BoardGame, Player
#-------------------------------------------------------------------------
'''
    Problem 1: TicTacToe and MiniMax AI player 
    In this problem, you will implement two different AI players for the TicTacToe game.
'''

#-------------------------------------------------------
class TicTacToe(BoardGame):
    '''
        TicTacToe game environment: the goal is to provide a platform for two AI players to play the game in turns and return the game result. 
    '''

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


#-------------------------------------------------------
''' 
    AI Player 1 (Random Player): Let's first implement the simplest player agent (Random Agent) to get familiar with TicTacToe AI.
'''
#-------------------------------------------------------
class RandomPlayer(Player):
    '''
        Random player: it chooses a random valid move at each step of the game. 
        This player is the simplest AI agent for the tic-tac-toe game. 
        It is also the foundation of Monte-Carlo Sampling which we will need to use later.
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
                s: the current state of the game, which is an object of GameState class.
                 (1)s.b represents the current setting of the board, an integer matrix of shape 3 by 3 (TicTacToe) or 8 by 8 (Othello).
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X" player. 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by "O" player.
                    For example, in TicTacToe, the following game state 
                     | X |   | O |
                     | O | X |   |
                     | X |   | O |
                    is represented as the following numpy matrix
                    s.b= [[ 1 , 0 ,-1 ],
                          [-1 , 1 , 0 ],
                          [ 1 , 0 ,-1 ]]

                 (2) s.x: who's turn in this step of the game, x=1 if "X" player's turn; x=-1 if "O" player's turn 
           Outputs:
                r: the row number of the next move, an integer scalar.
                c: the column number of the next move, an integer scalar.
            For example, in the above example, the valid moves are the empty grid cells: 
            (r=0,c=1) --- the first row, second column 
            (r=1,c=2) --- the second row, the third column 
            (r=2,c=1) --- the third row , the second column
            The random player should randomly choose one of the valid moves. 
            Hint: for different board games, you can always use get_valid_moves() function in the game.py to get the list of valid moves.
            Hint: you could solve this problem using 3 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # find all valid moves in the current game state
        m=g.get_valid_moves(s)
        # randomly choose one valid move
        i = np.random.randint(len(m))
        r,c = m[i]
        #########################################
        return r,c

    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_choose_a_move' in the terminal.  '''



    #-----------------------------------------------
    ''' 
        Great job!

        ------
        DEMO 1: If your code has passed all the above tests, now you can play TicTacToe game with the AI (RandomPlayer).  
        ------
        
        # INSTALL 
        In order to run the demo, you need to install pygame package:
        In the terminal, type the following:
        pip3 install pygame

        # RUN DEMO:
        type `python3 demo1.py' in the terminal
    '''
    #-----------------------------------------------
    ''' DEMO 2: You can also play the Othello game with the AI (RandomPlayer) by typing `python3 demo2.py' in the terminal.  '''
    #-----------------------------------------------
    #-----------------------------------------------
    ''' DEMO 3: You can also play the GO game with the AI (RandomPlayer) by typing `python3 demo3.py' in the terminal.  '''
    #-----------------------------------------------




 
#-----------------------------------------------
''' 
    AI Player 2 (MiniMax Player): Now let's implement the MiniMax agent for the game.
    The goal of this agent is to find the optimal (best) move for the current game state.
    MiniMax player will build a fully expanded search tree, where each tree node corresponds to a game state.
    The root of the tree is the current game state.
    Then we compute the score (value) of each node recursively using minimax algorithm.
    Finally, the MiniMax agent will choose the child node with the largest value as the next move. 

    For example, suppose the current game state is:
          s.b=[[ 0, 1, 1],
               [ 0,-1,-1],
               [ 1,-1, 1]]
    and it's "O" player's turn (s.x=-1).
    Then the search tree will be: 
   |-------------------
   |Root Node:   
   |  s.b=[[ 0, 1, 1],
   |       [ 0,-1,-1],
   |       [ 1,-1, 1]]     -- the game state in the node
   |  s.x=-1               -- it's "O" player's turn at this step of the game
   |    p= None            -- The root node has no parent node
   |    m= None            -- the move it takes from parent node to this node (no parent node) 
   |    c=[Child_Node_A, Child_Node_B] -- list of children nodes
   |    v=-1               -- The value of the game state:  
   |                            assuming both players follows their optimal moves,
   |                            from this game state,  "O" player will win (-1).
   |----------------------------
      |Child Node A:  ---- If O player chooses the move (r=0,c=0)
      |  s.b=[[-1, 1, 1],
      |       [ 0,-1,-1],
      |       [ 1,-1, 1]] -- the game state in the node
      |  s.x= 1           -- it's "X" player's turn 
      |    p= Root_Node   -- The parent node is the root node
      |    m= (0,0)       -- from parent node, took the move (r=0, c=0) to reach this node
      |    c=[Grand Child Node C] -- list of children nodes
      |    v= 1           -- The value of the game state:  
      |                            assuming both players follows their optimal moves,
      |                            from this game state,  "X" player will win (1).
      |----------------------------
          |Grand Child Node A1: ---- If X player chooses the move (r=1, c=0)
          |  s.b=[[-1, 1, 1],
          |       [ 1,-1,-1],
          |       [ 1,-1, 1]]    -- the game state in the node
          |  s.x=-1              -- it's "O" player's turn 
          |    p= Child_Node_B   -- The parent node is the child node B
          |    m= (1,0)          -- from parent node, took the move (r=1,c=0) to reach this node
          |    c=[] -- list of children nodes, no child node because the game has ended
          |    v= 0               -- The score of the game state:  
          |                          Terminal node, the game ends with a Tie (0).
      |------------------------------ 
      |Child Node B:  ---- If O player chooses the move (r=1,c=0)
      |  s.b=[[ 0, 1, 1],
      |       [-1,-1,-1],
      |       [ 1,-1, 1]]     -- the game state in the node
      |  s.x= 1               -- it's "X" player's turn in this step of the game
      |    p= Root_Node       -- The parent node is the root node
      |    m= (1,0)           -- from parent node, took the move (r=1,c=0) to reach this node
      |    c=[] -- list of children nodes, no child node because the game has ended
      |    v=-1               -- The value of the game state:  
      |                           Terminal node, the game ends: O player won (-1) 
      |--------------------------


        The tree looks like this:
 
                          |--> Child Node A (v=0) |--> Grand Child Node A1 (v=0)
        Root Node(v=-1)-->| 
                          |--> Child Node B (v=-1) 

        In this example, the two children nodes have values as v=0 (child A) and v=-1 (child B). 
        The "O" player will choose the child node with the smallest value as the next move.
        In this example, the smallest value is Child Node B (v=-1), so the optimal next move is (r=1, c=0)


------------------------------------------------------------
MiniMax is a search-tree-based  methods. 
Now let's implement tree nodes first.  Then we can connect the nodes into a search tree.
------------------------------------------------------------
'''
class Node:
    '''
        Search Tree Node. This is a base/general class of search tree node, which can be used later by all search tree based methods, such as MiniMax, Monte-Carlo Tree Search.

        List of Attributes: 
            s: the current state of the game, 
                s.b is an integer matrix  (shape: 3 by 3 in TictacToe, 8 by 8 in Othello). 
                s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                s.x: who's turn in this step of the game (if X player: x=1, or if O player: x=-1)
            p: the parent node of this node 
            m: the move that it takes from the parent node to reach this node.  m is a tuple (r,c), r:row of move, c:column of the move 
            c: a python list of all the children nodes of this node 
    '''
    def __init__(self,s, p=None,c=None,m=None,v=None):
        self.s = s # the current state of the game
        self.p= p # the parent node of the current node
        self.m=m # the move that it takes from the parent node to reach this node. 
                 # m is a tuple (r,c), r:row of move, c:column of the move 
        self.c=[] # a list of children nodes
        self.v=v # the value of the node (X player will win:1, tie: 0, lose: -1)

# ----------------------------------------------
class MMNode(Node):
    '''
        MiniMax Search Tree Node. 

        List of Attributes: 
            s: the current state of the game, 
                s.b is an integer matrix  (shape: 3 by 3 in TictacToe, 8 by 8 in Othello). 
                s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                s.x: who's turn in this step of the game (if X player: x=1, or if O player: x=-1)
            p: the parent node of this node 
            m: the move that it takes from the parent node to reach this node.  m is a tuple (r,c), r:row of move, c:column of the move 
            c: a python list of all the children nodes of this node 
            v: the value of the node (-1, 0, or 1). We assume both players are choosing optimal moves, then this value v represents the best score that "X" player can achieve (1: win, 0:tie, -1: loss)
    '''
    def __init__(self,s,p=None,c=None,m=None,v=None):
        super(MMNode, self).__init__(s,p=p,c=c,m=m,v=v)

    # ----------------------------------------------
    def expand(self,g):
        '''
            In order to build a search tree, we first need to implement an elementary operation:  
            Expand the current tree node by adding one layer of children nodes.
            Add one child node for each valid next move.
        Input:
            self: the node to be expanded
            g: the game environment being played, such as TicTacToe or Othello. 

        For example, in TicTacToe, if the current node (BEFORE expanding) is like:
       |-------------------
       |Current Node:   
       |  s.b=[[ 0, 1,-1],
       |       [ 0,-1, 1],
       |       [ 0, 1,-1]]     -- the game state in the node
       |  s.x= 1               -- it's "X" player's turn in this step of the game
       |    p= None           
       |    m= None            
       |    c=[] -- no children node
       |    v= None               
       |-------------------

        There are 3 valid next moves from the current game state.
        AFTER expanding this node, we add three children nodes to the current node.
        The tree looks like this after being expanded:

                            |--> Child Node A
           Current Node --> |--> Child Node B 
                            |--> Child Node C 

        Here are the details of the tree (attributes of each tree node):
       |-------------------
       |Current Node:   
       |  s.b=[[ 0, 1,-1],
       |       [ 0,-1, 1],
       |       [ 0, 1,-1]]     
       |  s.x= 1        -- it's "X" player's turn in this step of the game  
       |    p= None           
       |    m= None            
       |    c=[Child_A, Child_B, Child_C] -- Three children nodes are created and added here
       |    v= None               
       |-------------------------------
               |Child Node A:   
               |  s.b=[[ 1, 1,-1],
               |       [ 0,-1, 1],
               |       [ 0, 1,-1]]     
               |  s.x=-1            -- it's "O" player's turn in this step of the game 
               |    p= Current_Node -- The parent node of this node is "Current_Node" 
               |    m= (0,0)        -- The move it takes from parent node 
               |                         to this node: first row (0), first column (0) 
               |    c=[] -- this node has not been expanded yet 
               |    v= None               
               |-----------------------
               |Child Node B:   
               |  s.b=[[ 0, 1,-1],
               |       [ 1,-1, 1],
               |       [ 0, 1,-1]]     
               |  s.x=-1            -- it's "O" player's turn in this step of the game 
               |    p= Current_Node -- The parent node of this node is "Current_Node" 
               |    m= (1,0)        -- The move it takes from parent node 
               |                        to this node: second row (1), first column (0) 
               |    c=[] -- this node has not been expanded yet 
               |    v= None               
               |-----------------------
               |Child Node C:   
               |  s.b=[[ 0, 1,-1],
               |       [ 0,-1, 1],
               |       [ 1, 1,-1]]     
               |  s.x=-1            -- it's "O" player's turn in this step of the game 
               |    p= Current_Node -- The parent node of this node is "Current_Node" 
               |    m= (2,0)        -- The move it takes from parent node 
               |                        to this node: third row (2), first column (0) 
               |    c=[] -- this node has not been expanded yet 
               |    v= None               
               |-----------------------
            Hint: you could use g.get_move_state_pairs() function to compute all the next moves and next game states in the game.
            Hint: you could solve this problem using 4 lines of code.
            Hint: in Othello game, given a parent node p and its child node c,  p.x and c.x NOT necessarily have opposite sign.  When there is no valid move for one player, that player will give up the move, so in this case the p.x and c.x can be the same.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # get the list of valid next move-state pairs from the current game state
        p=g.get_move_state_pairs(self.s)
         
        # expand the node with one level of children nodes 
        for m, s in p:
            # for each next move m and game state s, create a child node
            c = MMNode(s,p=self, m=m)
            # append the child node the child list of the current node 
            self.c.append(c)
        #########################################

    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_expand' in the terminal.  '''

 
    # ----------------------------------------------
    def build_tree(self,g):
        '''
        Given a tree node (the current state of the game), build a fully-grown search tree, which includes all the possible future game states in the tree.  
        Note, in this function, we don't need to compute the values of the nodes, just leave them as None. 
        We will compute them later in compute_v().    

        Input:
            self: the root node (current game state) to be expanded into a fully-grown tree
            g: the game environment being played, such as TicTacToe or Othello. 

        For example, in TicTacToe, the current node is (BEFORE building tree)
        |-------------------
        |Current Node:   
        |  s.b=[[ 0, 1,-1],
        |       [ 0,-1, 1],
        |       [ 0, 1,-1]]     -- the game state in the node
        |  s.x= 1               -- it's "X" player's turn in this step of the game
        |    p= None           
        |    m= None            
        |    c=[] -- list of children nodes
        |    v= None               
        |-------------------

        AFTER expanding this node, we have a tree as follows:
        The tree looks like this after being expanded:

                                                 |--> Grand Child Node A1 |--> Great Grand Child A11
                            |--> Child Node A -->| 
                            |                    |--> Grand Child Node A2
                            |
                            |                    |--> Grand Child Node B1 
           Current Node --> |--> Child Node B -->| 
                            |                    |--> Grand Child Node B2 
                            |
                            |                    |--> Grand Child Node C1 
                            |--> Child Node C -->| 
                                                 |--> Grand Child Node C2 |--> Great Grand Child C21

       Each node of the tree represents a possible future game state.
       Here are the detailed attribute values of tree nodes:
       --------------------
       |Current Node:   
       |  s.b=[[ 0, 1,-1],
       |       [ 0,-1, 1],
       |       [ 0, 1,-1]]     
       |  s.x= 1        -- it's "X" player's turn in this step of the game  
       |    p= None           
       |    m= None            
       |    c=[Child_A, Child_B, Child_C] -- Three children nodes are created and added here
       |    v= None               
       |-------------------------------
           |Child Node A:   
           |  s.b=[[ 1, 1,-1],
           |       [ 0,-1, 1],
           |       [ 0, 1,-1]]     
           |  s.x=-1               -- it's "O" player's turn in this step of the game 
           |    p= Current_Node    -- The parent node of this node is "Current_Node" 
           |    m= (0,0)           -- The move it takes to from parent node 
           |                           to this node is first row (0), first column (0) 
           |    c=[Grand_Child_A, Grand_Child_B] -- Two children nodes 
           |    v= None               
           |-------------------------------
                   |Grand Child Node A1:   
                   |  s.b=[[ 1, 1,-1],
                   |       [-1,-1, 1],
                   |       [ 0, 1,-1]]     
                   |  s.x= 1            -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_A -- The parent node of this node is "Child Node A" 
                   |    m= (1,0)        -- The move it takes from parent node 
                   |                         to this node: second row (1),first column (0) 
                   |    c=[Great_Grand_Child_A11] -- one child node
                   |    v= None       
                   |--------------------------------
                           |Great Grand Child Node A11:   
                           |  s.b=[[ 1, 1,-1],
                           |       [-1,-1, 1],
                           |       [ 1, 1,-1]]     
                           |  s.x=-1             -- it's "O" player's turn in this step of the game 
                           |    p= Grand_Child_Node_A1  -- The parent node of this node 
                           |    m= (2,0)         -- The move from parent node 
                           |                        to this node is third row (2),first column (0) 
                           |    c=[] -- Terminal node (no child) 
                           |    v= None       
                   -------------------------
                   |Grand Child Node A2:   
                   |  s.b=[[ 1, 1,-1],
                   |       [ 0,-1, 1],
                   |       [-1, 1,-1]]     
                   |  s.x= 1            -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_A -- The parent node of this node is "Child Node A" 
                   |    m= (2,0)        -- The move it takes from parent node 
                   |                        to this node: third row (2),first column (0) 
                   |    c=[] -- terminal node (game ends), no child node 
                   |    v= None    
           |-----------------------
           |Child Node B:   
           |  s.b=[[ 0, 1,-1],
           |       [ 1,-1, 1],
           |       [ 0, 1,-1]]     
           |  s.x=-1            -- it's "O" player's turn in this step of the game 
           |    p= Current_Node -- The parent node of this node is "Current_Node" 
           |    m= (1,0)        -- The move it takes from parent node to this node
           |    c=[] -- this node has not been expanded yet 
           |    v= None               
           |--------------------------------
                   |Grand Child Node B1:   
                   |  s.b=[[-1, 1,-1],
                   |       [ 1,-1, 1],
                   |       [ 0, 1,-1]]     
                   |  s.x= 1             -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_B  -- The parent node of this node 
                   |    m= (0,0)         -- The move it takes from parent node to this node
                   |    c=[]             -- Terminal node (no child)
                   |    v= None       
                   -------------------------
                   |Grand Child Node B2:   
                   |  s.b=[[ 0, 1,-1],
                   |       [ 1,-1, 1],
                   |       [-1, 1,-1]]     
                   |  s.x= 1             -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_B  -- The parent node of this node 
                   |    m= (2,0)         -- The move it takes from parent node to this node
                   |    c=[] -- Terminal node (no child) 
                   |    v= None    
           |--------------------------------
           |Child Node C:   
           |  s.b=[[ 0, 1,-1],
           |       [ 0,-1, 1],
           |       [ 1, 1,-1]]     
           |  s.x=-1               -- it's "O" player's turn in this step of the game 
           |    p= Current_Node    -- The parent node of this node is "Current_Node" 
           |    m= (2,0)           -- The move it takes to from parent node to this node
           |    c=[] -- this node has not been expanded yet 
           |    v= None               
           |-------------------------------
                   |Grand Child Node C1:   
                   |  s.b=[[-1, 1,-1],
                   |       [ 0,-1, 1],
                   |       [ 1, 1,-1]]     
                   |  s.x= 1               -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_A    -- The parent node of this node is "Child Node A" 
                   |    m= (0,0)           -- The move it takes to from parent node to this node 
                   |    c=[] -- game ends, no child 
                   |    v= None       
                   -------------------------
                   |Grand Child Node C2:   
                   |  s.b=[[ 0, 1,-1],
                   |       [-1,-1, 1],
                   |       [ 1, 1,-1]]     
                   |  s.x= 1             -- it's "X" player's turn in this step of the game 
                   |    p= Child_Node_A  -- The parent node of this node is "Child Node A" 
                   |    m= (1,0)         -- The move it takes from parent node to this node
                   |    c=[Great_Grand_Child_C21] -- one child node 
                   |    v= None  
                   |--------------------------------
                           |Great Grand Child Node C21:   
                           |  s.b=[[ 1, 1,-1],
                           |       [-1,-1, 1],
                           |       [ 1, 1,-1]]     
                           |  s.x=-1            -- it's "O" player's turn in this step of the game 
                           |    p= Grand_Child_Node_C2  -- The parent node of this node 
                           |    m= (0,0)        -- The move  from parent node to this node 
                           |    c=[] -- Terminal node (no child) 
                           |    v= None     
                           |------------------------

        Hint: you could use recursion to build the tree and solve this problem using 4 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        
        # if the game in the current state has not ended yet 
        if g.check_game(self.s) is None:
            #expand the current node by one-level of children nodes
            self.expand(g)
            # recursion: for each child node, call build_tree() function 
            # to build a subtree rooted from each child node
            for c in self.c:
                c.build_tree(g)
 
        #########################################


    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_build_tree' in the terminal.  '''



    # ----------------------------------------------
    def compute_v(self,g):
        '''
            Given a fully-built tree, compute optimal values of the all the nodes in the tree using minimax algorithm
            Here we assume that the whole search-tree is fully grown, but no value on any node has been computed yet before calling this function.
           
        Input:
            self: the current node to compute value 
            g: the game environment being played, such as TicTacToe or Othello. 

        MinMax Algorithm: 
        The optimal value of a tree node is defined as follows:
        (1) if the node is a terminal node, the value of the node is the game result (1, -1 or 0)
        (2) if the node has children nodes, which means that it is not a terminal node, and the game has not ended yet 
                (2.1) if it is X player's turn in the current node:
                        the value of the node is maximum value of all the children nodes' values.
                (2.2) if it is O player's turn in the current node:
                        the value of the node is minimum value of all the children nodes' values.

        For example, the current game state is
        |-------------------
        |Current Node:   
        |  s.b=[[ 1,-1, 1],
        |       [ 0, 0, 0],
        |       [ 0, 0,-1]]     -- the game state in the node
        |  s.x= 1               -- it's "X" player's turn in this step of the game
        |    p= None           
        |    m= None            
        |    c=[] -- list of children nodes
        |    v= None               
        |-------------------
        
        The search tree will have 5 levels of children nodes.
        The first two levels of the tree looks like this:

                            |--> Child Node A -->|--> Grand Child Node A1 
                            |     1,-1, 1        |--> Grand Child Node A2 
                            |     1, 0, 0        |--> Grand Child Node A3
                            |     0, 0,-1        |--> Grand Child Node A4 
                            |
                            |--> Child Node B -->|--> Grand Child Node B1 
                            |     1,-1, 1        |--> Grand Child Node B2 
                            |     0, 1, 0        |--> Grand Child Node B3 
                            |     0, 0,-1        |--> Grand Child Node B4 
                            |
          Current Node -->  |--> Child Node C -->|--> Grand Child Node C1 
           1,-1, 1          |     1,-1, 1        |--> Grand Child Node C2 
           0, 0, 0          |     0, 0, 1        |--> Grand Child Node C3 
           0, 0,-1          |     0, 0,-1        |--> Grand Child Node C4 
                            |
                            |--> Child Node D -->|--> Grand Child Node D1 
                            |     1,-1, 1        |--> Grand Child Node D2 
                            |     0, 0, 0        |--> Grand Child Node D3  
                            |     1, 0,-1        |--> Grand Child Node D4 
                            |
                            |--> Child Node E -->|--> Grand Child Node E1 
                                  1,-1, 1        |--> Grand Child Node E2 
                                  0, 0, 0        |--> Grand Child Node E3  
                                  0, 1,-1        |--> Grand Child Node E4 

        If we finish computing the values of all the Grand Children nodes, we have: 
        
                                 (O's turn)             
                            |--> Child Node A -->|--> Grand Child Node A1 (v=1) 
                            |     1,-1, 1        |--> Grand Child Node A2 (v=1) 
                            |     1, 0, 0        |--> Grand Child Node A3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node A4 (v=1) 
                            |
                            |--> Child Node B -->|--> Grand Child Node B1 (v=1) 
                            |     1,-1, 1        |--> Grand Child Node B2 (v=1) 
                            |     0, 1, 0        |--> Grand Child Node B3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node B4 (v=1) 
           (X's turn)       |
          Current Node -->  |--> Child Node C -->|--> Grand Child Node C1 (v=0) 
           1,-1, 1          |     1,-1, 1        |--> Grand Child Node C2 (v=0) 
           0, 0, 0          |     0, 0, 1        |--> Grand Child Node C3 (v=0) 
           0, 0,-1          |     0, 0,-1        |--> Grand Child Node C4 (v=-1) 
                            |
                            |--> Child Node D -->|--> Grand Child Node D1 (v=1) 
                            |     1,-1, 1        |--> Grand Child Node D2 (v=1) 
                            |     0, 0, 0        |--> Grand Child Node D3 (v=1)  
                            |     1, 0,-1        |--> Grand Child Node D4 (v=1) 
                            |
                            |--> Child Node E -->|--> Grand Child Node E1 (v=0) 
                                  1,-1, 1        |--> Grand Child Node E2 (v=0) 
                                  0, 0, 0        |--> Grand Child Node E3 (v=1)  
                                  0, 1,-1        |--> Grand Child Node E4 (v=0) 

        In Child Node A, it is "O" player's turn, so the value of Child Node A is the MINIMUM of all its children nodes' values: min(1,1,0,1) = 0
        Similarly, we can compute all the children nodes' (A,B,C,D).

                                 (O's turn)             
                            |--> Child Node A -->|--> Grand Child Node A1 (v=1) 
                            |     1,-1, 1 (v=0)  |--> Grand Child Node A2 (v=1) 
                            |     1, 0, 0        |--> Grand Child Node A3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node A4 (v=1) 
                            |
                            |--> Child Node B -->|--> Grand Child Node B1 (v=1) 
                            |     1,-1, 1 (v=0)  |--> Grand Child Node B2 (v=1) 
                            |     0, 1, 0        |--> Grand Child Node B3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node B4 (v=1) 
           (X's turn)       |
          Current Node -->  |--> Child Node C -->|--> Grand Child Node C1 (v=0) 
           1,-1, 1          |     1,-1, 1 (v=-1) |--> Grand Child Node C2 (v=0) 
           0, 0, 0          |     0, 0, 1        |--> Grand Child Node C3 (v=0) 
           0, 0,-1          |     0, 0,-1        |--> Grand Child Node C4 (v=-1) 
                            |
                            |--> Child Node D -->|--> Grand Child Node D1 (v=1) 
                            |     1,-1, 1 (v=1)  |--> Grand Child Node D2 (v=1) 
                            |     0, 0, 0        |--> Grand Child Node D3 (v=1)  
                            |     1, 0,-1        |--> Grand Child Node D4 (v=1) 
                            |
                            |--> Child Node E -->|--> Grand Child Node E1 (v=0) 
                                  1,-1, 1 (v=1)  |--> Grand Child Node E2 (v=0) 
                                  0, 0, 0        |--> Grand Child Node E3 (v=1)  
                                  0, 1,-1        |--> Grand Child Node E4 (v=0) 

        Now the values of all the children nodes of the current node are ready, we can compute the value of the current node.
        In the current node, it is "X" player's turn, so the value of the current node is the MAXIMUM of all its children nodes' values: max(0,0,-1,1,0) = 1

                                 (O's turn)             
                            |--> Child Node A -->|--> Grand Child Node A1 (v=1) 
                            |     1,-1, 1 (v=0)  |--> Grand Child Node A2 (v=1) 
                            |     1, 0, 0        |--> Grand Child Node A3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node A4 (v=1) 
                            |
                            |--> Child Node B -->|--> Grand Child Node B1 (v=1) 
                            |     1,-1, 1 (v=0)  |--> Grand Child Node B2 (v=1) 
                            |     0, 1, 0        |--> Grand Child Node B3 (v=0) 
                            |     0, 0,-1        |--> Grand Child Node B4 (v=1) 
           (X's turn)       |
          Current Node -->  |--> Child Node C -->|--> Grand Child Node C1 (v=0) 
           1,-1, 1 (v=1)    |     1,-1, 1 (v=-1) |--> Grand Child Node C2 (v=0) 
           0, 0, 0          |     0, 0, 1        |--> Grand Child Node C3 (v=0) 
           0, 0,-1          |     0, 0,-1        |--> Grand Child Node C4 (v=-1) 
                            |
                            |--> Child Node D -->|--> Grand Child Node D1 (v=1) 
                            |     1,-1, 1 (v=1)  |--> Grand Child Node D2 (v=1) 
                            |     0, 0, 0        |--> Grand Child Node D3 (v=1)  
                            |     1, 0,-1        |--> Grand Child Node D4 (v=1) 
                            |
                            |--> Child Node E -->|--> Grand Child Node E1 (v=0) 
                                  1,-1, 1 (v=0)  |--> Grand Child Node E2 (v=0) 
                                  0, 0, 0        |--> Grand Child Node E3 (v=1)  
                                  0, 1,-1        |--> Grand Child Node E4 (v=0) 
        Hint: you could use recursion to compute the values of the current node recursively. 
              you could use 12 lines of code to solve this problem.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        # (1) if the game has already ended, the value of the node is the game result 
        e = g.check_game(self.s)
        if e is not None:
            self.v=e
            return 
        
        # (2) if the game has not ended yet: 
        #   (2.1)first compute values of all children nodes recursively by calling compute_v() in each child node
        v = []
        for c in self.c:
            c.compute_v(g)
            v.append(c.v)
        #   (2.2) now the values of all the children nodes are computed, let's compute the value of the current node:
        #       (2.2.1) if it is X player's turn, the value of the current node is the max of all children node's values 
        #       (2.2.2) if it is O player's turn, the value of the current node is the min of all children node's values 
        if self.s.x==1:
            self.v=max(v)
        else:
            self.v=min(v)
        #########################################

    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_compute_v' in the terminal.  '''



#-----------------------------------------------
''' 
    AI Player 2 (MinMax Player): Now let's implement the MinMax agent for the game.
    The goal of this agent is to find the optimal (best) move for the current game state.
    (1) Build Tree: we will first build a fully-grown search tree, where the root of the tree is the current game state.
    (2) Compute Node Values: Then we compute the value of each node recursively using MinMax algorithm.
    (3) Choose Optimal Next Move: the agent will choose the child node with the largest/smallest value as the next move.
            if the MinMax player is the "X" player in the game, it will choose the largest value among children nodes. 
            if the MinMax player is the "O" player in the game, it will choose the smallest value among children nodes. 
'''

#-------------------------------------------------------
class MiniMaxPlayer(Player):
    '''
        Minimax player, who choose optimal moves by searching the tree with min-max.  
    '''
    #----------------------------------------------
    # Let's first implement step (3): choose optimal next move
    def choose_optimal_move(self,n):
        '''
            Assume we have a fully-grown search tree, and the values of all nodes are already computed.
    
            (3) Choose Next Move: the agent will choose the child node with the largest/smallest value as the next move.
                if the MinMax player is the "X" player in the game, it will choose the largest value among children nodes. 
                if the MinMax player is the "O" player in the game, it will choose the smallest value among children nodes. 
    
           Inputs:
                n: the current node of the search tree, assuming the values in all nodes are already computed.
           Outputs:
                r: the row number of the optimal next move, an integer scalar with value 0, 1, or 2. 
                c: the column number of the optimal next move, an integer scalar with value 0, 1, or 2. 
    
            For example, suppose we have the following search tree (X player's turn):
                                    |--> Child Node A 
                                    |    |1,-1, 1|(v=0)  
                                    |    |1, 0, 0|(m=(1,0))       
                                    |    |0, 0,-1|       
                                    |
                                    |--> Child Node B 
                                    |    |1,-1, 1|(v=0)  
                                    |    |0, 1, 0|(m=(1,1))       
                                    |    |0, 0,-1|       
                   (X's turn)       |
                  Current Node -->  |--> Child Node C 
                  |1,-1, 1|(v=1)    |    |1,-1, 1|(v=-1) 
                  |0, 0, 0|         |    |0, 0, 1|(m=(1,2))       
                  |0, 0,-1|         |    |0, 0,-1|        
                                    |
                                    |--> Child Node D 
                                    |    |1,-1, 1|(v=1)  
                                    |    |0, 0, 0|(m=(2,0))         
                                    |    |1, 0,-1|       
                                    |
                                    |--> Child Node E 
                                         |1,-1, 1|(v=0)  
                                         |0, 0, 0|(m=(2,1))         
                                         |0, 1,-1|       
            The optimal next move will be child node with the largest value (Child Node D). 
            So in this example, the next move should be (r=2, c=0)
            Hint: you could solve this problem using 5 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE
        v = []
        for c in n.c:
            v.append(c.v)
        idx=np.argmax(np.array(v)*n.s.x)
        r,c = n.c[idx].m
        #########################################
        return r,c
    
        ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_choose_optimal_move' in the terminal.  '''


    # ----------------------------------------------
    def choose_a_move(self,g,s):
        '''
            The action function of the minimax player, which chooses next move in the game.  
            The goal of this agent is to find the optimal (best) move for the current game state.
            (1) Build Tree: we will first build a fully-grown search tree, where the root of the tree is the current game state.
            (2) Compute Node Values: Then we compute the value of each node recursively using MinMax algorithm.
            (3) Choose Next Move: the agent will choose the child node with the largest/smallest value as the next move.
                    if the MinMax player is the "X" player in the game, it will choose the largest value among children nodes. 
                    if the MinMax player is the "O" player in the game, it will choose the smallest value among children nodes. 
           Inputs:
                g: the game environment being played, such as TicTacToe or Othello. 
                s: the current state of the game, 
                    s.b is an integer matrix. 
                    s.b[i,j] = 0 denotes that the i-th row and j-th column is empty
                    s.b[i,j] = 1 denotes that the i-th row and j-th column is taken by "X". 
                    s.b[i,j] = -1 denotes that the i-th row and j-th column is taken by the "O".
                    s.x: the role of the player, 1 if you are the "X" player in the game
                         -1 if you are the "O" player in the game. 
           Outputs:
                r: the row number of the next move, an integer scalar with value 0, 1, or 2. 
                c: the column number of the next move, an integer scalar with value 0, 1, or 2. 
          Hint: you could solve this problem using 4 lines of code.
        '''
        #########################################
        ## INSERT YOUR CODE HERE

        # (1) build a search tree with the current game state as the root node
        n = MMNode(s)
        n.build_tree(g)

        # (2) compute values of all tree nodes
        n.compute_v(g) 
        
        # (3) choose the optimal next move
        r,c = self.choose_optimal_move(n)
        #########################################
        return r,c

    ''' TEST: Now you can test the correctness of your code above by typing `nosetests -v test1.py:test_minmax_choose_a_move' in the terminal.  '''



#--------------------------------------------

''' TEST Problem 1: 
        Now you can test the correctness of all the above functions by typing `nosetests -v test1.py' in the terminal.  

        If your code passed all the tests, you will see the following message in the terminal:
            ----------- Problem 1 (50 points in total)--------------------- ... ok
            (5 points) get_valid_moves() ... ok
            (5 points) check_game() ... ok
            (5 points) apply_a_move() ... ok
            (5 points) random choose_a_move() ... ok
            (5 points) expand ... ok
            (5 points) build_tree ... ok
            (5 points) compute_v() ... ok
            (5 points) choose_optimal_move() ... ok
            (10 points) minmax choose_a_move() ... ok
            ----------------------------------------------------------------------
            Ran 10 tests in 2.939s            
            OK
'''

#--------------------------------------------




#-----------------------------------------------
''' 
    Great job!
    DEMO 1: If your code has passed all the above tests, now you can play TicTacToe game with the AI (MiniMaxPlayer) 
    by typing `python3 demo1.py minimax' in the terminal.  
'''
#-----------------------------------------------
''' DEMO 2: Othello: Unfortunately, Othello is a larger game where the MiniMax method won't work. 
    In larger games, we will need sampling-based method, such as Monte-Carlo Tree Search in Problem 2'''
#-----------------------------------------------



