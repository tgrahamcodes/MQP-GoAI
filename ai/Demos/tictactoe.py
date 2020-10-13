import pygame
import numpy as np
import sys
import os
sys.pcath.append(os.path.abspath('..\\GoAI\\ai\\Players'))
from minimax import RandomPlayer, MiniMaxPlayer
from mcts import MCTSPlayer
sys.path.append(os.path.abspath('..\\GoAI\\ai'))
from game import TicTacToe 

'''
    This is a demo for TicTacToe game. You could play with the AI that you have built in minimax and mcts.
    # INSTALL 
    In order to run this demo, you need to install pygame package:
    In the terminal, type the following:
        pip3 install pygame
   
    # RUN A GAME 
    If you want to play with random player, you could type the following in the terminal:
        python3 tictactoe.py

    If you want to play with MiniMax player, you could type the following in the terminal:
        python3 tictactoe.py minimax

    If you want to play with MCTS player, you could type the following in the terminal:
        python3 tictactoe.py mcts
'''
screenSize = 400
margin = 35
gameSize = screenSize - (2 * margin)
lineSize = 5
backgroundColor = (0, 0, 0)

# load image for stones
x_img = pygame.image.load('..\\GOAI\\ai\\x.png')
o_img = pygame.image.load('..\\GOAI\\ai\\o.png')

#---------------------------------------------
def map_mouse_to_board(x, y):
    '''
        map the mouse to the grid on board
        Input:
            x: the x location of the mouse`
            y: the y location of the mouse`
        Outputs:
            row: row number
            column: column number
    '''
    if x < gameSize / 3 + margin: column = 0
    elif gameSize / 3+margin <= x < (gameSize / 3) * 2+margin: column = 1
    else: column = 2
    if y < gameSize / 3 + margin: row = 0
    elif gameSize / 3 + margin <= y < (gameSize / 3) * 2 + margin:row = 1
    else:row = 2
    return row, column


#---------------------------------------------
def draw_board(win,s):
    '''
        Draw the board based upon the game state
        Inputs:
            win: the window to draw in
            s: the game state
    '''
    for y in range(3):
        for x in range(3):
            picker = lambda xx,oo: xx if s[y][x] == 1 else oo if s[y][x] == -1 else pygame.Surface((0, 0))
            win.blit(picker(x_img, o_img), (x * (gameSize // 3) + margin + 17,15+ y * (gameSize // 3) + margin) )

#---------------------------------------------
def draw_lines(win):
    '''
        Draw the board lines
        Inputs:
            win: the window to draw in
    '''
    # Vertical lines
    pygame.draw.line(win, (255, 255, 255), (margin + gameSize // 3, margin),
                     (margin + gameSize // 3, screenSize - margin), lineSize)
    pygame.draw.line(win, (255, 255, 255), (margin + (gameSize // 3) * 2, margin),
                     (margin + (gameSize // 3) * 2, screenSize - margin), lineSize)
    # Horizontal lines
    pygame.draw.line(win, (255, 255, 255), (margin, margin + gameSize // 3), (screenSize - margin, margin + gameSize // 3),
                     lineSize)
    pygame.draw.line(win, (255, 255, 255), (margin, margin + (gameSize // 3) * 2),
                     (screenSize - margin, margin + (gameSize // 3) * 2), lineSize)

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
    s.fill((0,0,0))       
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
def draw_empty_board(win):
    # background
    win.fill(backgroundColor)
    # Draw the board
    draw_lines(win)

#---------------------------------------------
def init_screen():
    # initialize the game
    pygame.init()
    # start the display window
    win = pygame.display.set_mode((screenSize, screenSize))


    # set icon
    icon = pygame.image.load('..\\GOAI\\ai\\o.png')
    pygame.display.set_icon(icon)

    # Title
    pygame.display.set_caption("Tic Tac Toe")
    pygame.font.init()
    myFont = pygame.font.SysFont('Tahoma', gameSize // 3)

    # draw empty board
    draw_empty_board(win)

    return win


#---------------------------------------------
def run_a_game(p):
    '''
        Run a game
        Input:
            p: the AI player that you are playing with 
    '''
    win = init_screen()

    # initialize game state
    g = TicTacToe()
    s = g.initial_game_state()
    x = 1 # current turn (x player's turn)

    # create game tree if it does not exist, otherwise load it
    _ = p.choose_a_move(g, s)

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
                draw_empty_board(win)
                canPlay = True
                x=1 # X player's turn
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
            # print(mouseX)
            # print(mouseY)
            # convert to board grid (row,column)
            r, c = map_mouse_to_board(mouseX, mouseY)
            # if the move is valid 
            if g.check_valid_move(s,r,c):
                # update game state
                g.apply_a_move(s,r,c)
                x=s.x

                # draw the board
                draw_board(win,s.b)
                print("X player chooses:",str(r),str(c))

                # check if the game has ended already
                e = g.check_game(s) 
                if e is not None:
                    draw_result(win,e)
                    canPlay = False
                e=pygame.event.Event(pygame.USEREVENT)
                pygame.event.post(e)

        if event.type == pygame.USEREVENT and x== -1 and canPlay: # computer's turn to choose a move
            r,c = p.choose_a_move(g,s)
            # if the move is valid 
            assert g.check_valid_move(s,r,c)
            # update game state
            g.apply_a_move(s,r,c)
            x=s.x
            # draw the board
            draw_board(win,s.b)
            print("O player chooses:",str(r),str(c))

            # check if the game has ended already
            e = g.check_game(s) 
            if e is not None:
                draw_result(win,e)
                canPlay = False
    
        # update the UI display
        pygame.display.update()

if __name__ == "__main__":
    if len(sys.argv)>1:
        arg=sys.argv[1]
        if arg=="mcts": # play with MCTS player
            p =MCTSPlayer()
            print('Now you are playing with Monte-Carlo Tree Search Player!')
        elif arg=="minimax": # player with MiniMax player
            p = MiniMaxPlayer()
            print('Now you are playing with MiniMax Player!')
        else:
            assert False # Incorrect AI name
    else:
        p= RandomPlayer() # default: random player
        print('Now you are playing with Random Player!')
    run_a_game(p)

