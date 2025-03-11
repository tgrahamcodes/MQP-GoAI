import pygame
import sys
import os
from colorama import init, Fore, Back, Style
init(autoreset=True)  # Initialize colorama
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"  # Hide Pygame welcome message
from pathlib import Path
from ..Players.minimax import RandomPlayer, MiniMaxPlayer
from ..Players.mcts import MCTSPlayer
from ..Players.qfcnn import QFcnnPlayer
from ..Players.policynn import PolicyNNPlayer
from ..Players.valuenn import ValueNNPlayer
from ..game import TicTacToe 
import numpy as np
from . import game_utils as gu

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
x_file = Path(__file__).parents[1].joinpath('x.PNG')
o_file = Path(__file__).parents[1].joinpath('o.PNG')
x_img = pygame.image.load(str(x_file))
o_img = pygame.image.load(str(o_file))

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
    if x < gameSize / 3 + margin:
        column = 0
    elif gameSize / 3 + margin <= x < (gameSize / 3) * 2 + margin:
        column = 1
    else: 
        column = 2
    if y < gameSize / 3 + margin:
        row = 0
    elif gameSize / 3 + margin <= y < (gameSize / 3) * 2 + margin:
        row = 1
    else:
        row = 2
    return row, column


#---------------------------------------------
def draw_board(win,g,s):
    '''
        Draw the board based upon the game state
        Inputs:
            win: the window to draw in
            g: the game object
            s: the game state
    '''
    for y in range(3):
        for x in range(3):
            def picker(xx, oo):
                return xx if s.b[y][x] == 1 else oo if s.b[y][x] == -1 else pygame.Surface((0, 0))
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
    icon = pygame.image.load(str(o_file))
    pygame.display.set_icon(icon)

    # Title
    pygame.display.set_caption("Tic Tac Toe")
    pygame.font.init()

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
    # _ = p.choose_a_move(g, s)

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
                gu.print_move("X", r, c, s.b, "tictactoe")

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
            gu.print_move("O", r, c, s.b, "tictactoe")
    
        # update the UI display
        pygame.display.update()

def print_board_state(board):
    """Print the current board state in color"""
    print("\n     0   1   2  ")
    print("   +---+---+---+")
    for i in range(3):
        print(f" {i} |", end="")
        for j in range(3):
            if board[i][j] == 1:
                print(f" {Fore.GREEN}X{Style.RESET_ALL} |", end="")
            elif board[i][j] == -1:
                print(f" {Fore.RED}O{Style.RESET_ALL} |", end="")
            else:
                print("   |", end="")
        print("\n   +---+---+---+")

def print_move(player, r, c, board):
    """Print moves in a more visible format with colors"""
    if player == "X":
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎮 Your move (X): row={r}, col={c}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.RED}🤖 Computer's move (O): row={r}, col={c}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    
    print_board_state(board)

if __name__ == "__main__":
    # Clear the terminal
    gu.clear_screen()
    
    if len(sys.argv)>1:
        arg=sys.argv[1]
        if arg=="mcts": # play with MCTS player
            p = MCTSPlayer()
            ai_type = "Monte-Carlo Tree Search AI"
        elif arg=="qfcnn": # play with QFCNN player
            p = QFcnnPlayer()
            ai_type = "Q-Learning Neural Network AI"
        elif arg=="policy": # play with PolicyNN player
            p = PolicyNNPlayer()
            ai_type = "Policy Neural Network AI"
        elif arg=="value": # play with ValueNN player
            p = ValueNNPlayer()
            ai_type = "Value Neural Network AI"
        elif arg=="minimax": # play with MiniMax player
            p = MiniMaxPlayer()
            ai_type = "MiniMax AI"
        else: # play with Random player
            p = RandomPlayer()
            ai_type = "Random AI"
    else: # play with Random player by default
        p = RandomPlayer()
        ai_type = "Random AI"
    
    gu.print_welcome("TicTacToe", ai_type)
    
    # Print empty board at start
    gu.print_tictactoe_board(np.zeros((3, 3)))
    
    run_a_game(p)

