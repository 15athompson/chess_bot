import chess
import random

def evaluate_board(board):
    """
    Evaluates the current state of the chess board from the perspective of white.

    Args:
        board: The chess board object.

    Returns:
        int: A positive score indicates an advantage for white,
             a negative score indicates an advantage for black,
             and a score of 0 indicates a neutral position.
    """

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is not None:
            if piece.color == chess.WHITE:
                score += piece_values[piece.piece_type]
            else:
                score -= piece_values[piece.piece_type]
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    """
    Minimax algorithm with alpha-beta pruning to determine the best move.

    Args:
        board: The chess board object.
        depth: The maximum depth to search the game tree.
        alpha: The best score for the maximizing player found so far.
        beta: The best score for the minimizing player found so far.
        maximizing_player: True if it's the maximizing player's turn, False otherwise.

    Returns:
        tuple: The best move found and its corresponding score.
    """

    if depth == 0 or board.is_game_over():
        return None, evaluate_board(board)

    best_move = None
    if maximizing_player:
        best_score = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            _, score = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_move, best_score
    else:
        best_score = float('inf')
        for move in board.legal_moves:
            board.push(move)
            _, score = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            if score < best_score:
                best_score = score
                best_move = move
            beta = min(beta, best_score)
            if beta <= alpha:
                break
        return best_move, best_score

def get_ai_move(board, depth):
    """
    Gets the AI's move using the minimax algorithm.

    Args:
        board: The chess board object.
        depth: The depth for the minimax search.

    Returns:
        chess.Move: The best move found by the AI.
    """

    move, _ = minimax(board, depth, float('-inf'), float('inf'), board.turn)
    return move

def play_game():
    """
    Plays a game of chess between two AI players.

    The function initializes a chess board and alternates between the two AIs making moves
    until the game is over. After each move, the board state is printed.

    At the end of the game, the result is determined and printed.

    Returns:
        None
    """

    board = chess.Board()
    depth = 3  # Adjust the depth for stronger/weaker AI

    while not board.is_game_over():
        move = get_ai_move(board, depth)
        board.push(move)
        print(f"Move: {move}")
        print(board)
        print("\n")

    result = board.result()
    if result == "1-0":
        print("White wins!")
    elif result == "0-1":
        print("Black wins!")
    else:
        print("It's a draw!")

if __name__ == "__main__":
    play_game()
