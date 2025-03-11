import os
from colorama import init, Fore, Back, Style
init(autoreset=True)  # Initialize colorama

def clear_screen():
    """Clear the terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_welcome(game_name, ai_type):
    """Print welcome message with colors"""
    print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Welcome to {game_name}!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}\n")
    print(f"{Fore.MAGENTA}🎮 You are playing against the {ai_type}!{Style.RESET_ALL}")
    
    print(f"\n{Fore.WHITE}Controls:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}- Click on a cell to make your move{Style.RESET_ALL}")
    print(f"{Fore.WHITE}- Press F to restart the game{Style.RESET_ALL}")
    print(f"{Fore.WHITE}- Press ESC to quit{Style.RESET_ALL}")
    
    print(f"\n{Fore.YELLOW}Game starting...{Style.RESET_ALL}")
    print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}\n")

def print_tictactoe_board(board):
    """Print the TicTacToe board state in color"""
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

def print_go_board(board, size):
    """Print the Go board state in color"""
    print("\n     " + "  ".join(str(i).rjust(2) for i in range(size)))
    print("   +" + "---+" * size)
    for i in range(size):
        print(f" {str(i).rjust(2)} |", end="")
        for j in range(size):
            if board[i][j] == 1:
                print(f" {Fore.GREEN}●{Style.RESET_ALL} |", end="")
            elif board[i][j] == -1:
                print(f" {Fore.RED}○{Style.RESET_ALL} |", end="")
            else:
                print("   |", end="")
        print(f"\n   +{'---+' * size}")

def print_othello_board(board):
    """Print the Othello board state in color"""
    print("\n     " + "  ".join(str(i).rjust(2) for i in range(8)))
    print("   +" + "---+" * 8)
    for i in range(8):
        print(f" {str(i).rjust(2)} |", end="")
        for j in range(8):
            if board[i][j] == 1:
                print(f" {Fore.GREEN}●{Style.RESET_ALL} |", end="")
            elif board[i][j] == -1:
                print(f" {Fore.RED}○{Style.RESET_ALL} |", end="")
            else:
                print("   |", end="")
        print(f"\n   +{'---+' * 8}")

def print_move(player, r, c, board, game_type="tictactoe"):
    """Print moves in a more visible format with colors"""
    if player == "X":
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎮 Your move (X): row={r}, col={c}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.RED}🤖 Computer's move (O): row={r}, col={c}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    
    if game_type == "tictactoe":
        print_tictactoe_board(board)
    elif game_type == "go":
        print_go_board(board, len(board))
    elif game_type == "othello":
        print_othello_board(board)

def print_pass_move(player):
    """Print pass move in Go game"""
    if player == "X":
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎮 Your move: PASS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}")
        print(f"{Fore.RED}🤖 Computer's move: PASS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'=' * 40}{Style.RESET_ALL}") 