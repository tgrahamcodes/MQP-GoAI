import pygame
import numpy as np
import sys
from problem1 import RandomPlayer 
from game import GO 

'''
    This is a demo for GO game. You could play with the AI that you have built in problem 1.
   
    # RUN A GAME 
    If you want to play with random player, you could type the following in the terminal:
        python3 demo3.py
'''

Board_SIZE = 19 # 19 or 10 or 5
screenSize = 700 
margin = 700//Board_SIZE
gameSize = screenSize - (2 * margin)
backgroundColor = (220, 220,150)
xplayerColor = (0,0,0)
oplayerColor = (255,255,255)
xplayer_nextmoveColor = (50,50,50)
oplayer_nextmoveColor = (200,200,200)
MAX_GAME_LENGTH = Board_SIZE*Board_SIZE*3

#---------------------------------------------
def draw_lines(win,n=Board_SIZE-1):
    '''
        Draw the board lines
        Inputs:
            win: the window to draw in
            n: number of row / columns in the board. 
    '''

    w = gameSize//n 
    # Vertical lines
    for i in range(n+1):
        pygame.draw.line(win,(0, 0, 0), 
                        (margin + w*i, margin),
                        (margin + w*i, screenSize - margin), 
                        2)
    # Horizontal lines
    for i in range(n+1):
        pygame.draw.line(win, (0, 0, 0), 
                        (margin,                margin + w*i), 
                        (screenSize - margin,   margin + w*i),
                        2)
    if n == 18:
        # draw points
        a = margin + w*3
        b = margin + w*(19-4)
        c = margin + w*9
        pygame.draw.circle(win, (0, 0, 0), [a, a],w//7)
        pygame.draw.circle(win, (0, 0, 0), [a, b],w//7)
        pygame.draw.circle(win, (0, 0, 0), [b, a],w//7)
        pygame.draw.circle(win, (0, 0, 0), [b, b],w//7)
        pygame.draw.circle(win, (0, 0, 0), [a, c],w//7)
        pygame.draw.circle(win, (0, 0, 0), [c, a],w//7)
        pygame.draw.circle(win, (0, 0, 0), [b, c],w//7)
        pygame.draw.circle(win, (0, 0, 0), [c, b],w//7)
        pygame.draw.circle(win, (0, 0, 0), [c, c],w//7)

#---------------------------------------------
def map_mouse_to_board(x, y,n=Board_SIZE-1):
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
    row = (y-margin+w//2)//w
    column= (x-margin+w//2)//w 
    return row, column


#---------------------------------------------
def draw_board(win,g,s,n=Board_SIZE-1):
    '''
        Draw the board based upon the game state
        Inputs:
            win: the window to draw in
            s: the game state
    '''
    win.fill(backgroundColor)
    draw_lines(win)
    w = gameSize//n
    for i in range(n+1):
        for j in range(n+1):
            a=margin+ j*w+1
            b=margin+ i*w+1
            if s.b[i,j]==1:
                pygame.draw.circle(win, xplayerColor, [a, b],w//2-2 )
            elif s.b[i,j]==-1:
                pygame.draw.circle(win, oplayerColor, [a, b],w//2-2)


#---------------------------------------------
def draw_result(win,e):
    '''
        Draw the game result on the screen 
        Input:
            win: the window to draw in
            e: the result of the game: 1: x player wins, -1: O player wins, 0: draw
    '''
    s = pygame.Surface((700,700))  
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
    icon = pygame.image.load('o.png')
    pygame.display.set_icon(icon)

    # Title
    pygame.display.set_caption("GO, press key 'p' to pass")
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
    g = GO(board_size=Board_SIZE)
    g.M=MAX_GAME_LENGTH # maximum game length
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
            # press P button (pass the current move)
            if event.key == pygame.K_p:
                g.apply_a_move(s,None,None)
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
                print("X player chooses: PASS")
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
            if r is None:
                print("O player chooses: PASS")
            else:
                print("O player chooses:",str(r),str(c))
    
        # update the UI display
        pygame.display.update()

if __name__ == "__main__":
    if len(sys.argv)>1:
        arg=sys.argv[1]
        # play with MCTS player
        from problem2 import MCTSPlayer
        p =MCTSPlayer(n_iter = 300)
        print('Now you are playing with Monte-Carlo Tree Search Player!')
    else:
        p= RandomPlayer() # default: random player
        print('Now you are playing with Random Player!')
    run_a_game(p)


