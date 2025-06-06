import tensorflow as tf
import numpy as np
import random

class ChessBoard:
    def __init__(self, fen=None):
        self.board = [['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r'],
                      ['p', 'p', 'p', 'p', 'p', 'p', 'p', 'p'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['.', '.', '.', '.', '.', '.', '.', '.'],
                      ['P', 'P', 'P', 'P', 'P', 'P', 'P', 'P'],
                      ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']]
        self.turn = 'w'
        self.move_history = []  # Track move history for undo functionality
        self.board_history = []  # Track board states for undo functionality
        if fen:
            self.load_fen(fen)

    def load_fen(self, fen):
        parts = fen.split()
        rows = parts[0].split('/')
        for i, row in enumerate(rows):
            col = 0
            for char in row:
                if char.isdigit():
                    col += int(char)
                else:
                    self.board[i][col] = char
                    col += 1
        self.turn = 'w' if parts[1] == 'w' else 'b'

    def to_fen(self):
        fen = []
        for row in self.board:
            empty = 0
            row_fen = ''
            for piece in row:
                if piece == '.':
                    empty += 1
                else:
                    if empty > 0:
                        row_fen += str(empty)
                        empty = 0
                    row_fen += piece
            if empty > 0:
                row_fen += str(empty)
            fen.append(row_fen)
        return '/'.join(fen) + f" {self.turn} KQkq - 0 1"

    def make_move(self, move):
        # Save current board state and move to history before making the move
        self.save_state(move)

        from_col, from_row = ord(move[0]) - ord('a'), 8 - int(move[1])
        to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
        piece = self.board[from_row][from_col]

        # Handle pawn promotion
        if piece.lower() == 'p':
            # Check if pawn reaches the end of the board
            if (piece == 'P' and to_row == 0) or (piece == 'p' and to_row == 7):
                # Promote to queen by default
                piece = 'Q' if piece == 'P' else 'q'

        self.board[from_row][from_col] = '.'
        self.board[to_row][to_col] = piece
        self.turn = 'b' if self.turn == 'w' else 'w'

    def is_game_over(self):
        # Get legal moves that don't leave king in check
        legal_moves = self.get_legal_moves_safe()
        if not legal_moves:
            return True

        # Check for insufficient material (simplified)
        pieces = []
        for row in self.board:
            for piece in row:
                if piece != '.':
                    pieces.append(piece.lower())

        # If only kings remain, it's a draw
        if len(pieces) == 2 and pieces.count('k') == 2:
            return True

        return False

    def get_result(self):
        if not self.is_game_over():
            return '*'

        # Get legal moves that don't leave king in check
        legal_moves = self.get_legal_moves_safe()
        if not legal_moves:
            # No legal moves - check if it's checkmate or stalemate
            if self.is_in_check(self.turn):
                # King is in check and no legal moves = checkmate
                if self.turn == 'w':
                    return '0-1'  # Black wins
                else:
                    return '1-0'  # White wins
            else:
                # King not in check but no legal moves = stalemate
                return '1/2-1/2'

        return '*'

    def get_legal_moves(self):
        """Get all pseudo-legal moves (may leave king in check)"""
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (piece.isupper() and self.turn == 'w') or (piece.islower() and self.turn == 'b'):
                    moves.extend(self.get_piece_moves(row, col))
        return moves

    def get_legal_moves_safe(self):
        """Get all legal moves that don't leave king in check"""
        pseudo_legal_moves = self.get_legal_moves()
        legal_moves = []

        for move in pseudo_legal_moves:
            if self.is_legal_move(move):
                legal_moves.append(move)

        return legal_moves

    def get_piece_moves(self, row, col):
        piece = self.board[row][col].lower()
        if piece == 'p':
            return self.get_pawn_moves(row, col)
        elif piece == 'r':
            return self.get_rook_moves(row, col)
        elif piece == 'n':
            return self.get_knight_moves(row, col)
        elif piece == 'b':
            return self.get_bishop_moves(row, col)
        elif piece == 'q':
            return self.get_queen_moves(row, col)
        elif piece == 'k':
            return self.get_king_moves(row, col)
        return []

    def get_pawn_moves(self, row, col):
        moves = []
        direction = -1 if self.turn == 'w' else 1
        start_row = 6 if self.turn == 'w' else 1

        # Forward move
        if 0 <= row + direction < 8:
            if self.board[row + direction][col] == '.':
                moves.append(f"{chr(col + ord('a'))}{8-row}{chr(col + ord('a'))}{8-(row+direction)}")

                # Double move from starting position
                if row == start_row and 0 <= row + 2 * direction < 8:
                    if self.board[row + 2 * direction][col] == '.':
                        moves.append(f"{chr(col + ord('a'))}{8-row}{chr(col + ord('a'))}{8-(row+2*direction)}")

        # Diagonal captures
        for dc in [-1, 1]:
            new_row, new_col = row + direction, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target != '.' and ((self.turn == 'w' and target.islower()) or (self.turn == 'b' and target.isupper())):
                    moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")

        return moves

    def get_rook_moves(self, row, col):
        moves = []
        # Horizontal and vertical moves
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if self.board[new_row][new_col] == '.':
                        moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
                    else:
                        # Can capture opponent piece
                        target_piece = self.board[new_row][new_col]
                        if (self.turn == 'w' and target_piece.islower()) or (self.turn == 'b' and target_piece.isupper()):
                            moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
                        break
                else:
                    break
        return moves

    def get_knight_moves(self, row, col):
        moves = []
        knight_moves = [(-2, -1), (-2, 1), (-1, -2), (-1, 2), (1, -2), (1, 2), (2, -1), (2, 1)]
        for dr, dc in knight_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == '.' or ((self.turn == 'w' and target.islower()) or (self.turn == 'b' and target.isupper())):
                    moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
        return moves

    def get_bishop_moves(self, row, col):
        moves = []
        # Diagonal moves
        directions = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for dr, dc in directions:
            for i in range(1, 8):
                new_row, new_col = row + dr * i, col + dc * i
                if 0 <= new_row < 8 and 0 <= new_col < 8:
                    if self.board[new_row][new_col] == '.':
                        moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
                    else:
                        # Can capture opponent piece
                        target_piece = self.board[new_row][new_col]
                        if (self.turn == 'w' and target_piece.islower()) or (self.turn == 'b' and target_piece.isupper()):
                            moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
                        break
                else:
                    break
        return moves

    def get_queen_moves(self, row, col):
        # Queen combines rook and bishop moves
        return self.get_rook_moves(row, col) + self.get_bishop_moves(row, col)

    def get_king_moves(self, row, col):
        moves = []
        king_moves = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for dr, dc in king_moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < 8 and 0 <= new_col < 8:
                target = self.board[new_row][new_col]
                if target == '.' or ((self.turn == 'w' and target.islower()) or (self.turn == 'b' and target.isupper())):
                    moves.append(f"{chr(col + ord('a'))}{8-row}{chr(new_col + ord('a'))}{8-new_row}")
        return moves

    def save_state(self, move):
        """Save current board state and move for undo functionality"""
        # Deep copy the current board state
        board_copy = [row[:] for row in self.board]
        self.board_history.append({
            'board': board_copy,
            'turn': self.turn,
            'move': move
        })
        self.move_history.append(move)

    def undo_move(self):
        """Undo the last move and return True if successful, False if no moves to undo"""
        if not self.board_history:
            return False

        # Restore the previous board state
        previous_state = self.board_history.pop()
        self.move_history.pop()

        self.board = previous_state['board']
        self.turn = previous_state['turn']
        return True

    def can_undo(self):
        """Check if there are moves that can be undone"""
        return len(self.board_history) > 0

    def find_king(self, color):
        """Find the king position for the given color"""
        king_piece = 'K' if color == 'w' else 'k'
        for row in range(8):
            for col in range(8):
                if self.board[row][col] == king_piece:
                    return (row, col)
        return None

    def is_square_attacked(self, row, col, by_color):
        """Check if a square is attacked by the given color"""
        # Temporarily switch turn to check attacks
        original_turn = self.turn
        self.turn = by_color

        # Get all possible moves for the attacking color
        for r in range(8):
            for c in range(8):
                piece = self.board[r][c]
                if (piece.isupper() and by_color == 'w') or (piece.islower() and by_color == 'b'):
                    moves = self.get_piece_moves(r, c)
                    for move in moves:
                        # Parse move to get target square
                        target_col = ord(move[2]) - ord('a')
                        target_row = 8 - int(move[3])
                        if target_row == row and target_col == col:
                            self.turn = original_turn
                            return True

        self.turn = original_turn
        return False

    def is_in_check(self, color):
        """Check if the king of the given color is in check"""
        king_pos = self.find_king(color)
        if king_pos is None:
            return False

        opponent_color = 'b' if color == 'w' else 'w'
        return self.is_square_attacked(king_pos[0], king_pos[1], opponent_color)

    def is_legal_move(self, move):
        """Check if a move is legal (doesn't leave king in check)"""
        # Make the move temporarily
        from_col, from_row = ord(move[0]) - ord('a'), 8 - int(move[1])
        to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])

        original_piece = self.board[from_row][from_col]
        captured_piece = self.board[to_row][to_col]

        # Make the move
        self.board[from_row][from_col] = '.'
        self.board[to_row][to_col] = original_piece

        # Check if king is in check after the move
        in_check = self.is_in_check(self.turn)

        # Undo the move
        self.board[from_row][from_col] = original_piece
        self.board[to_row][to_col] = captured_piece

        return not in_check

class ChessAI:
    def __init__(self):
        self.model = self.create_model()
        self.piece_values = {
            'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0,
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0
        }
        self.nodes_searched = 0
        self.max_nodes = 50000  # Limit search to prevent hanging
        import time
        self.start_time = 0
        self.max_time = 10.0  # Maximum 10 seconds per move

        # Piece-Square Tables for positional evaluation
        self.pawn_table = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [50, 50, 50, 50, 50, 50, 50, 50],
            [10, 10, 20, 30, 30, 20, 10, 10],
            [5,  5, 10, 25, 25, 10,  5,  5],
            [0,  0,  0, 20, 20,  0,  0,  0],
            [5, -5,-10,  0,  0,-10, -5,  5],
            [5, 10, 10,-20,-20, 10, 10,  5],
            [0,  0,  0,  0,  0,  0,  0,  0]
        ]

        self.knight_table = [
            [-50,-40,-30,-30,-30,-30,-40,-50],
            [-40,-20,  0,  0,  0,  0,-20,-40],
            [-30,  0, 10, 15, 15, 10,  0,-30],
            [-30,  5, 15, 20, 20, 15,  5,-30],
            [-30,  0, 15, 20, 20, 15,  0,-30],
            [-30,  5, 10, 15, 15, 10,  5,-30],
            [-40,-20,  0,  5,  5,  0,-20,-40],
            [-50,-40,-30,-30,-30,-30,-40,-50]
        ]

        self.bishop_table = [
            [-20,-10,-10,-10,-10,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5, 10, 10,  5,  0,-10],
            [-10,  5,  5, 10, 10,  5,  5,-10],
            [-10,  0, 10, 10, 10, 10,  0,-10],
            [-10, 10, 10, 10, 10, 10, 10,-10],
            [-10,  5,  0,  0,  0,  0,  5,-10],
            [-20,-10,-10,-10,-10,-10,-10,-20]
        ]

        self.rook_table = [
            [0,  0,  0,  0,  0,  0,  0,  0],
            [5, 10, 10, 10, 10, 10, 10,  5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [-5,  0,  0,  0,  0,  0,  0, -5],
            [0,  0,  0,  5,  5,  0,  0,  0]
        ]

        self.queen_table = [
            [-20,-10,-10, -5, -5,-10,-10,-20],
            [-10,  0,  0,  0,  0,  0,  0,-10],
            [-10,  0,  5,  5,  5,  5,  0,-10],
            [-5,  0,  5,  5,  5,  5,  0, -5],
            [0,  0,  5,  5,  5,  5,  0, -5],
            [-10,  5,  5,  5,  5,  5,  0,-10],
            [-10,  0,  5,  0,  0,  0,  0,-10],
            [-20,-10,-10, -5, -5,-10,-10,-20]
        ]

        self.king_table = [
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-30,-40,-40,-50,-50,-40,-40,-30],
            [-20,-30,-30,-40,-40,-30,-30,-20],
            [-10,-20,-20,-20,-20,-20,-20,-10],
            [20, 20,  0,  0,  0,  0, 20, 20],
            [20, 30, 10,  0,  0, 10, 30, 20]
        ]

        # Opening book - common good opening moves
        self.opening_book = {
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': ['e2e4', 'd2d4', 'g1f3', 'c2c4'],
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1': ['e7e5', 'c7c5', 'e7e6', 'c7c6'],
            'rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1': ['d7d5', 'g8f6', 'c7c5', 'e7e6'],
            'rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2': ['g1f3', 'f1c4', 'd2d3', 'b1c3'],
            'rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq d6 0 2': ['c2c4', 'g1f3', 'e2e3', 'b1c3']
        }

    def create_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(64,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def board_to_input(self, board):
        input_data = np.zeros(64, dtype=np.float32)
        for i, row in enumerate(board.board):
            for j, piece in enumerate(row):
                if piece != '.':
                    input_data[i*8 + j] = self.piece_values[piece] / 9  # Normalize the value
        return input_data.reshape(1, -1)

    def get_piece_square_value(self, piece, row, col):
        """Get positional value for a piece at given position"""
        piece_type = piece.lower()

        # For black pieces, flip the row (mirror the table)
        if piece.islower():
            row = 7 - row

        if piece_type == 'p':
            return self.pawn_table[row][col]
        elif piece_type == 'n':
            return self.knight_table[row][col]
        elif piece_type == 'b':
            return self.bishop_table[row][col]
        elif piece_type == 'r':
            return self.rook_table[row][col]
        elif piece_type == 'q':
            return self.queen_table[row][col]
        elif piece_type == 'k':
            return self.king_table[row][col]
        return 0

    def evaluate_position(self, board):
        """Evaluate position from current player's perspective (positive = good for current player)"""
        # Use reasonable piece values in centipawns (1 pawn = 100)
        piece_values_cp = {
            'p': 100, 'n': 320, 'b': 330, 'r': 500, 'q': 900, 'k': 0,
            'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0
        }

        white_score = 0
        black_score = 0

        # Count material for both sides
        for row in range(8):
            for col in range(8):
                piece = board.board[row][col]
                if piece != '.':
                    base_value = piece_values_cp[piece.upper()]

                    # Add small positional bonus (clamped to reasonable range)
                    positional_bonus = self.get_piece_square_value(piece, row, col)
                    positional_bonus = max(-30, min(30, positional_bonus))  # Limit to ±30

                    total_value = base_value + positional_bonus

                    if piece.isupper():  # White pieces
                        white_score += total_value
                    else:  # Black pieces
                        black_score += total_value

        # Calculate base material difference (White perspective)
        material_diff = white_score - black_score

        # Add small bonuses/penalties for game state
        bonus = 0

        # Check penalties (moderate)
        if board.is_in_check('w'):
            bonus -= 50  # White in check is bad
        if board.is_in_check('b'):
            bonus += 50  # Black in check is good for White

        # Center control (small bonus)
        center_squares = [(3, 3), (3, 4), (4, 3), (4, 4)]
        for row, col in center_squares:
            piece = board.board[row][col]
            if piece != '.':
                if piece.isupper():
                    bonus += 10  # White controls center
                else:
                    bonus -= 10  # Black controls center

        # Total score from White's perspective
        white_perspective_score = material_diff + bonus

        # Convert to current player's perspective
        if board.turn == 'w':
            final_score = white_perspective_score  # Positive = good for White
        else:
            final_score = -white_perspective_score  # Positive = good for Black

        # Clamp to reasonable range for display
        final_score = max(-2000, min(2000, final_score))

        return final_score

        # Score interpretation for users:
        # +100 = up a pawn (good position)
        # +300 = up a minor piece (winning)
        # +500 = up a rook (very winning)
        # +900 = up a queen (completely winning)

    def train_self_play(self, num_games=100):
        print(f"Starting training with {num_games} games...")
        for game_num in range(num_games):
            if game_num % 10 == 0:
                print(f"Training game {game_num + 1}/{num_games}")

            board = ChessBoard()
            game_history = []
            move_count = 0
            max_moves = 100  # Prevent infinite games

            while not board.is_game_over() and move_count < max_moves:
                input_data = self.board_to_input(board)
                legal_moves = board.get_legal_moves_safe()  # Use safe legal moves

                if not legal_moves:
                    break  # No legal moves available

                # Add some randomness to training (exploration)
                if random.random() < 0.3:  # 30% random moves for exploration
                    move = random.choice(legal_moves)
                else:
                    move = self.get_best_move(board)
                    if move is None:
                        move = random.choice(legal_moves)

                game_history.append((input_data, self.evaluate_position(board)))
                board.make_move(move)
                move_count += 1

            # Determine game result
            result = board.get_result()
            if result == "1-0":
                reward = 1
            elif result == "0-1":
                reward = -1
            else:
                reward = 0

            # Train the model on this game
            for input_data, evaluation in reversed(game_history):
                target = reward + 0.9 * evaluation / 20  # Discount factor of 0.9
                self.model.fit(input_data, np.array([target]), verbose=0)
                reward = target  # For the next iteration

        print(f"Training completed! Trained on {num_games} games.")

    def quiescence_search(self, board, alpha, beta, maximizing_player, depth=0):
        """Simplified quiescence search with depth limit"""
        # Limit quiescence depth to prevent infinite recursion
        if depth > 3:
            return self.evaluate_position(board)

        stand_pat = self.evaluate_position(board)

        if maximizing_player:
            if stand_pat >= beta:
                return beta
            alpha = max(alpha, stand_pat)
        else:
            if stand_pat <= alpha:
                return alpha
            beta = min(beta, stand_pat)

        # Only consider captures in quiescence search
        legal_moves = board.get_legal_moves_safe()
        capture_moves = []
        for move in legal_moves[:10]:  # Limit moves to check
            to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
            if board.board[to_row][to_col] != '.':
                capture_moves.append(move)

        if not capture_moves:
            return stand_pat

        # Order captures by captured piece value (only top 3)
        capture_moves.sort(key=lambda m: abs(self.piece_values.get(board.board[8 - int(m[3])][ord(m[2]) - ord('a')], 0)), reverse=True)
        capture_moves = capture_moves[:3]

        if maximizing_player:
            for move in capture_moves:
                temp_board = ChessBoard(board.to_fen())
                temp_board.make_move(move)
                score = self.quiescence_search(temp_board, alpha, beta, False, depth + 1)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return alpha
        else:
            for move in capture_moves:
                temp_board = ChessBoard(board.to_fen())
                temp_board.make_move(move)
                score = self.quiescence_search(temp_board, alpha, beta, True, depth + 1)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return beta

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Minimax algorithm with alpha-beta pruning and time/node limits"""
        import time

        # Check time and node limits
        self.nodes_searched += 1
        if (self.nodes_searched > self.max_nodes or
            time.time() - self.start_time > self.max_time):
            return self.evaluate_position(board)

        if depth == 0:
            return self.evaluate_position(board)

        if board.is_game_over():
            return self.evaluate_position(board)

        legal_moves = board.get_legal_moves_safe()
        if not legal_moves:
            # No legal moves - game over
            if board.is_in_check(board.turn):
                # Checkmate
                return -999999 if maximizing_player else 999999
            else:
                # Stalemate
                return 0

        # Order moves for better pruning (but limit to first 10 moves for speed)
        ordered_moves = self.order_moves(board, legal_moves)[:10]

        if maximizing_player:
            max_eval = float('-inf')
            for move in ordered_moves:
                temp_board = ChessBoard(board.to_fen())
                temp_board.make_move(move)
                eval_score = self.minimax(temp_board, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return max_eval
        else:
            min_eval = float('inf')
            for move in ordered_moves:
                temp_board = ChessBoard(board.to_fen())
                temp_board.make_move(move)
                eval_score = self.minimax(temp_board, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            return min_eval

    def order_moves(self, board, moves):
        """Order moves for better alpha-beta pruning (captures first, then others)"""
        captures = []
        non_captures = []

        for move in moves:
            to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
            if board.board[to_row][to_col] != '.':
                captures.append(move)
            else:
                non_captures.append(move)

        # Sort captures by value of captured piece (highest first)
        captures.sort(key=lambda m: abs(self.piece_values.get(board.board[8 - int(m[3])][ord(m[2]) - ord('a')], 0)), reverse=True)

        return captures + non_captures

    def get_best_move(self, board, difficulty=1.0):
        """Fast, reliable AI that won't hang"""
        import time
        start_time = time.time()

        legal_moves = board.get_legal_moves_safe()
        if not legal_moves:
            return None

        print(f"AI evaluating {len(legal_moves)} legal moves...")

        # Check opening book first
        current_fen = board.to_fen()
        if current_fen in self.opening_book:
            book_moves = self.opening_book[current_fen]
            valid_book_moves = [move for move in book_moves if move in legal_moves]
            if valid_book_moves:
                chosen_move = random.choice(valid_book_moves)
                print(f"AI using opening book move: {chosen_move}")
                return chosen_move

        # Fast evaluation approach - no deep search to avoid hanging
        best_move = None
        best_score = float('-inf') if board.turn == 'w' else float('inf')

        # Evaluate each move quickly
        for move in legal_moves[:15]:  # Limit to top 15 moves for speed
            temp_board = ChessBoard(board.to_fen())
            temp_board.make_move(move)

            # Quick evaluation
            score = self.evaluate_position(temp_board)

            # Bonus for captures
            to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
            captured_piece = board.board[to_row][to_col]
            if captured_piece != '.':
                capture_value = self.piece_values[captured_piece.upper()]
                score += capture_value if board.turn == 'w' else -capture_value

            # Bonus for getting out of check
            if board.is_in_check(board.turn) and not temp_board.is_in_check(board.turn):
                score += 500 if board.turn == 'w' else -500

            # Bonus for putting opponent in check
            opponent_color = 'b' if board.turn == 'w' else 'w'
            if temp_board.is_in_check(opponent_color):
                score += 100 if board.turn == 'w' else -100

            # Update best move
            if board.turn == 'w':
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

            # Time limit check
            if time.time() - start_time > 3.0:  # Max 3 seconds
                break

        # Fallback if no move found
        if best_move is None:
            best_move = self.get_quick_move(board, legal_moves)

        elapsed_time = time.time() - start_time
        print(f"AI selected move: {best_move} (time: {elapsed_time:.2f}s, score: {best_score:.1f})")
        return best_move

    def get_quick_move(self, board, legal_moves):
        """Quick move selection for fallback"""
        # Prioritize captures
        for move in legal_moves:
            to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
            if board.board[to_row][to_col] != '.':
                return move

        # If no captures, prioritize center moves
        center_moves = []
        for move in legal_moves:
            to_col, to_row = ord(move[2]) - ord('a'), 8 - int(move[3])
            if 2 <= to_row <= 5 and 2 <= to_col <= 5:  # Center area
                center_moves.append(move)

        if center_moves:
            return random.choice(center_moves)

        # Otherwise, random legal move
        return random.choice(legal_moves)