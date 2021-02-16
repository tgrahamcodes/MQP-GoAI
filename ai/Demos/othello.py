import pygame
import numpy as np
import sys
from pathlib import Path
from Players.minimax import RandomPlayer 
from Players.mcts import MCTSPlayer
from Players.qfcnn import QFcnnPlayer
from Players.policynn import PolicyNNPlayer
from Players.valuenn import ValueNNPlayer
from game import Othello 

'''
    This is a demo for Othello game. You could play with the AI that you have built in mcts.
   
    # RUN A GAME 
    If you want to play with random player, you could type the following in the terminal:
        python3 othello.py

    If you want to play with MCTS player, you could type the following in the terminal:
        python3 othello.py mcts
'''

screenSize = 400
margin = 35
gameSize = screenSize - (2 * margin)
backgroundColor = (0, 150, 150)
xplayerColor = (0,0,0)
oplayerColor = (255,255,255)
xplayer_nextmoveColor = (50,50,50)
oplayer_nextmoveColor = (200,200,200)

#---------------------------------------------
def draw_lines(win,n=8):
    '''
        Draw the board lines
        Inputs:
            win: the window to draw in
            n: number of row / columns in the board. 
    '''

    w = gameSize//n 
    # Vertical lines
    for i in range(n+1):
        pygame.draw.line(win,(255, 255, 255), 
                        (margin + w*i, margin),
                        (margin + w*i, screenSize - margin), 
                        2)
    # Horizontal lines
    for i in range(n+1):
        pygame.draw.line(win, (255, 255, 255), 
                        (margin,                margin + w*i), 
                        (screenSize - margin,   margin + w*i),
                        2)

#---------------------------------------------
def map_mouse_to_board(x, y,n=8):
    '''
        map the mouse to the grid on board
        Input:
            x: the x location of the mouse
            y: the y location of the mouse
            n: number of row / columns in the board. 
        Outputs:
            row: row number
            column: column number
    '''
    w = gameSize//n 
    row = (y-margin)//w
    column= (x-margin)//w
    return row, column


#---------------------------------------------
def draw_board(win,g,s,n=8):
    '''
        Draw the board based upon the game state
        Inputs:
            win: the window to draw in
            s: the game state
    '''
    win.fill(backgroundColor)
    draw_lines(win)
    w = gameSize//n
    for i in range(n):
        for j in range(n):
            a=margin+ j*w+w//2 +1
            b=margin+ i*w+w//2 +1
            if s.b[i,j]==1:
                pygame.draw.circle(win, xplayerColor, [a, b],w//2-2 )
            elif s.b[i,j]==-1:
                pygame.draw.circle(win, oplayerColor, [a, b],w//2-2)
    m=g.get_valid_moves(s)
    for r,c in m:
        a=margin+ c*w+w//2+1
        b=margin+ r*w+w//2+1
        if s.x==1:
            pygame.draw.circle(win, xplayer_nextmoveColor, [a, b],w//7 )
        else:
            pygame.draw.circle(win, oplayer_nextmoveColor, [a, b],w//7 )
        


#---------------------------------------------
def draw_result(win,e):
    '''
        Draw the game result on the screen 
        Input:
            win: the window to draw in
            e: the result of the game: 1: x player wins, -1: O player wins, 0: draw
    '''
    s = pygame.Surface((400,400))  
    s.set_alpha(230)           
    s.fill(backgroundColor)       
    win.blit(s, (0,0))
    myFont = pygame.font.SysFont('Verdana', 50)
    myFont2 = pygame.font.SysFont('Verdana', 20)
    if e==-1:
        text_surface = myFont.render("Computer won!", False, (255, 255, 255))
        size = myFont.size("Computer won!")
    elif e==0:
        text_surface = myFont.render("Draw!", False, (255, 255, 255))
        size = myFont.size("Draw!")
    else:
        text_surface = myFont.render("You won!", False, (255, 255, 255))
        size = myFont.size("You won!")

    text_surface2 = myFont2.render("press (F) to play again!", False, (255, 255, 255))
    size2 =myFont2.size("press (F) to play again!")
    win.blit(text_surface, (0.5*screenSize-0.5*size[0], screenSize // 2 - screenSize // 10))
    win.blit(text_surface2, (0.5*screenSize-0.5*size2[0], screenSize // 2 - screenSize // 10+100))

#---------------------------------------------
def draw_empty_board(win,g,s):
    # background
    win.fill(backgroundColor)
    # Draw the board
    draw_lines(win)
    draw_board(win,g,s)

#---------------------------------------------
def init_screen():
    # initialize the game
    pygame.init()
    # start the display window
    win = pygame.display.set_mode((screenSize, screenSize))


    # set icon
    o_file = Path(__file__).parents[1].joinpath('o.PNG')
    icon = pygame.image.load(str(o_file))
    pygame.display.set_icon(icon)

    # Title
    pygame.display.set_caption("Othello")
    pygame.font.init()
    myFont = pygame.font.SysFont('Tahoma', gameSize // 3)

    return win


#---------------------------------------------
def run_a_game(p):
    '''
        Run a game
        Input:
            p: the AI player that you are playing with 
    '''

    # initialize game state
    g = Othello()
    win = init_screen()

    # initialize the game state
    s = g.initial_game_state()
    x = 1 # current turn (x player's turn)
    # draw empty board
    draw_empty_board(win,g,s)

    canPlay = True
    pygame.display.update()

    # run the game
    while True:
        event = pygame.event.wait()
        # close the window 
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        
        # Press Key 
        if event.type == pygame.KEYDOWN:
            # press F button (restart game)
            if event.key == pygame.K_f:
                s = g.initial_game_state()
                x=1 # X player's turn
                draw_empty_board(win,g,s)
                canPlay = True
                pygame.display.update()
            # press ESC button (exit game)
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                sys.exit()
    
        # Click Mouse
        if event.type == pygame.MOUSEBUTTONDOWN and canPlay and x==1:
            # Human player's turn to choose a move
            # get mouse position
            (mouseX, mouseY) = pygame.mouse.get_pos()
            # convert to board grid (row,column)
            r, c = map_mouse_to_board(mouseX, mouseY)
            # if the move is valid 
            if g.check_valid_move(s,r,c):
                # update game state
                g.apply_a_move(s,r,c)
                x=s.x
                # draw the board
                draw_board(win,g,s)
                # check if the game has ended already
                e = g.check_game(s) 
                if e is not None:
                    draw_result(win,e)
                    canPlay = False
                e=pygame.event.Event(pygame.USEREVENT)
                pygame.event.post(e)
                print("X player chooses:",str(r),str(c))

        if event.type == pygame.USEREVENT and x== -1 and canPlay: # computer's turn to choose a move
            r,c = p.choose_a_move(g,s)
            # if the move is valid 
            assert g.check_valid_move(s,r,c)
            # update game state
            g.apply_a_move(s,r,c)
            x=s.x
            # draw the board
            draw_board(win,g,s)
            # check if the game has ended already
            e = g.check_game(s) 
            if e is not None:
                draw_result(win,e)
                canPlay = False
            e=pygame.event.Event(pygame.USEREVENT)
            pygame.event.post(e)
            print("O player chooses:",str(r),str(c))
    
        # update the UI display
        pygame.display.update()

if __name__ == "__main__":
    if len(sys.argv)>1:
        arg=sys.argv[1]
        if arg=="mcts": # play with MCTS player
            p =MCTSPlayer()
            print('Now you are playing with Monte-Carlo Tree Search Player!')
        elif arg=="qfcnn": # player with QFcnn player
            p = QFcnnPlayer()
            print('Now you are playing with QFcnn Player!')
        elif arg=="policy": # player with PolicyNN player
            p = PolicyNNPlayer()
            print('Now you are playing with PolicyNN Player!')
        else:
            assert False # Incorrect AI name
    else:
        p= RandomPlayer() # default: random player
        print('Now you are playing with Random Player!')
    run_a_game(p)


