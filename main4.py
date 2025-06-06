import chess
import chess.engine
import random

def evaluate_board(board):
    # Piece values
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Simple material evaluation
    score = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
    
    # Add a small random factor to avoid repetitive play
    score += random.uniform(-10, 10)
    
    return score

def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board)
    
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, False)
            board.pop()
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval = minimax(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval

def minimax_ai(board, depth=3):
    best_move = None
    best_eval = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta, False)
        board.pop()
        if eval > best_eval:
            best_eval = eval
            best_move = move
    return best_move

def stockfish_ai(board, engine, time_limit=0.1):
    result = engine.play(board, chess.engine.Limit(time=time_limit))
    return result.move

def play_game(engine_path):
    board = chess.Board()
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    
    while not board.is_game_over():
        if board.turn == chess.WHITE:
            move = minimax_ai(board)
        else:
            move = stockfish_ai(board, engine)
        
        board.push(move)
        print(f"Move: {move}")
        print(board)
        print(f"Evaluation: {evaluate_board(board)}")
        print("\n")
    
    result = board.result()
    if result == "1-0":
        print("White (Minimax AI) wins!")
    elif result == "0-1":
        print("Black (Stockfish AI) wins!")
    else:
        print("It's a draw!")
    
    engine.quit()

if __name__ == "__main__":
    stockfish_path = "/path/to/stockfish"  # Update this with your Stockfish engine path
    play_game(stockfish_path)