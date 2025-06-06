import chess
import random

def random_ai(board):
    return random.choice(list(board.legal_moves))

def slightly_smarter_ai(board):
    legal_moves = list(board.legal_moves)
    for move in legal_moves:
        if board.is_capture(move):
            return move
    return random.choice(legal_moves)

def play_game():
    """
    Plays a game of chess between a random AI and a slightly smarter AI.

    The function initializes a chess board and alternates between the two AIs making moves
    until the game is over. After each move, the board state is printed.

    At the end of the game, the result is determined and printed.

    Returns:
        None
    """
    board = chess.Board()
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = random_ai(board)
        else:
            move = slightly_smarter_ai(board)
        
        board.push(move)
        print(f"Move: {move}")
        print(board)
        print("\n")
    
    result = board.result()
    if result == "1-0":
        print("White (Random AI) wins!")
    elif result == "0-1":
        print("Black (Slightly Smarter AI) wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()
