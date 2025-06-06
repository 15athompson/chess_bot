import chess
import chess.engine
import chess.polyglot
import random
import PySimpleGUI as sg
import threading

# Piece-Square Tables for positional evaluation
pst = {
    chess.PAWN: [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ],
    chess.KNIGHT: [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ],
    chess.BISHOP: [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -20,-10,-10,-10,-10,-10,-10,-20,
    ],
    chess.ROOK: [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10, 10, 10, 10, 10,  5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        0,  0,  0,  5,  5,  0,  0,  0
    ],
    chess.QUEEN: [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -5,  0,  5,  5,  5,  5,  0, -5,
        0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ],
    chess.KING: [
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -10,-20,-20,-20,-20,-20,-20,-10,
         20, 20,  0,  0,  0,  0, 20, 20,
         20, 30, 10,  0,  0, 10, 30, 20
    ]
}

def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    score = 0
    piece_values = {chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
                    chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 20000}

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            position_value = pst[piece.piece_type][square if piece.color else chess.square_mirror(square)]
            score += (value + position_value) if piece.color == chess.WHITE else -(value + position_value)

    # Pawn structure evaluation
    for file in range(8):
        white_pawns = len(board.pieces(chess.PAWN, chess.WHITE) & chess.BB_FILES[file])
        black_pawns = len(board.pieces(chess.PAWN, chess.BLACK) & chess.BB_FILES[file])
        score += (white_pawns - black_pawns) * 10  # Penalize doubled pawns

    # King safety
    white_king_square = board.king(chess.WHITE)
    black_king_square = board.king(chess.BLACK)
    score += len(board.attackers(chess.BLACK, white_king_square)) * -10
    score -= len(board.attackers(chess.WHITE, black_king_square)) * 10

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

def get_best_move(board, depth):
    best_move = None
    best_eval = float('-inf') if board.turn == chess.WHITE else float('inf')
    alpha = float('-inf')
    beta = float('inf')
    
    # Check opening book first
    with chess.polyglot.open_reader("path/to/opening/book.bin") as reader:
        opening_move = reader.get(board)
        if opening_move:
            return opening_move.move
    
    for move in board.legal_moves:
        board.push(move)
        eval = minimax(board, depth - 1, alpha, beta, board.turn != chess.WHITE)
        board.pop()
        if board.turn == chess.WHITE:
            if eval > best_eval:
                best_eval = eval
                best_move = move
            alpha = max(alpha, eval)
        else:
            if eval < best_eval:
                best_eval = eval
                best_move = move
            beta = min(beta, eval)
    return best_move

def stockfish_ai(board, engine, skill_level):
    engine.configure({"Skill Level": skill_level})
    result = engine.play(board, chess.engine.Limit(time=0.1))
    return result.move

class ChessGame:
    def __init__(self, engine_path):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.window = None

    def create_board_image(self):
        # Create a chessboard image using PySimpleGUI
        board_layout = []
        for i in range(8):
            row = []
            for j in range(8):
                square = chess.square(j, 7-i)
                piece = self.board.piece_at(square)
                color = '#B58863' if (i + j) % 2 else '#F0D9B5'
                if piece:
                    symbol = piece.symbol()
                    button = sg.Button(symbol, size=(2, 1), pad=(0, 0), button_color=('black', color), key=(i,j))
                else:
                    button = sg.Button('', size=(2, 1), pad=(0, 0), button_color=(color, color), key=(i,j))
                row.append(button)
            board_layout.append(row)
        return board_layout

    def create_window(self):
        board_image = self.create_board_image()
        layout = [
            [sg.Column(board_image)],
            [sg.Text('', size=(20, 1), key='-STATUS-')],
            [sg.Button('New Game'), sg.Button('Exit')]
        ]
        self.window = sg.Window('Chess Game', layout)

    def update_board(self):
        for i in range(8):
            for j in range(8):
                square = chess.square(j, 7-i)
                piece = self.board.piece_at(square)
                color = '#B58863' if (i + j) % 2 else '#F0D9B5'
                if piece:
                    symbol = piece.symbol()
                    self.window[(i,j)].update(symbol, button_color=('black', color))
                else:
                    self.window[(i,j)].update('', button_color=(color, color))

    def play(self):
        self.create_window()

        while True:
            event, values = self.window.read()
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break
            elif event == 'New Game':
                self.board.reset()
                self.update_board()
                self.window['-STATUS-'].update('Your turn (White)')
            elif isinstance(event, tuple):  # Square clicked
                i, j = event
                square = chess.square(j, 7-i)
                move = chess.Move.from_uci(f"{chess.SQUARE_NAMES[square]}{chess.SQUARE_NAMES[square]}")
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board()
                    if not self.board.is_game_over():
                        self.window['-STATUS-'].update('AI is thinking...')
                        self.window.refresh()
                        ai_move = get_best_move(self.board, depth=4)  # Increased depth
                        self.board.push(ai_move)
                        self.update_board()
                        self.window['-STATUS-'].update('Your turn')
                    else:
                        self.window['-STATUS-'].update('Game Over')
                else:
                    self.window['-STATUS-'].update('Invalid move. Try again.')

        self.window.close()
        self.engine.quit()

if __name__ == "__main__":
    stockfish_path = "/path/to/stockfish"  # Update this with your Stockfish engine path
    game = ChessGame(stockfish_path)
    game.play()