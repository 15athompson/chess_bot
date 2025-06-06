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
        self.board[from_row][from_col] = '.'
        self.board[to_row][to_col] = piece
        self.turn = 'b' if self.turn == 'w' else 'w'

    def is_game_over(self):
        # Check if there are any legal moves available
        legal_moves = self.get_legal_moves()
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

        # Check if current player has legal moves
        legal_moves = self.get_legal_moves()
        if not legal_moves:
            # No legal moves - could be checkmate or stalemate
            # For simplicity, we'll call it a draw
            return '1/2-1/2'

        return '*'

    def get_legal_moves(self):
        moves = []
        for row in range(8):
            for col in range(8):
                piece = self.board[row][col]
                if (piece.isupper() and self.turn == 'w') or (piece.islower() and self.turn == 'b'):
                    moves.extend(self.get_piece_moves(row, col))
        return moves

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
        if 0 <= row + direction < 8:
            if self.board[row + direction][col] == '.':
                moves.append(f"{chr(col + ord('a'))}{8-row}{chr(col + ord('a'))}{8-(row+direction)}")
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

class ChessAI:
    def __init__(self):
        self.model = self.create_model()
        self.piece_values = {
            'p': 1, 'n': 3, 'b': 3, 'r': 5, 'q': 9, 'k': 0,
            'P': -1, 'N': -3, 'B': -3, 'R': -5, 'Q': -9, 'K': 0
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

    def evaluate_position(self, board):
        score = 0
        for row in board.board:
            for piece in row:
                if piece != '.':
                    score += self.piece_values[piece]
        return score

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
                legal_moves = board.get_legal_moves()

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

    def get_best_move(self, board, difficulty=1.0):
        legal_moves = board.get_legal_moves()
        if not legal_moves:
            return None
        
        best_move = None
        best_score = float('-inf') if board.turn == 'w' else float('inf')

        for move in legal_moves:
            temp_board = ChessBoard(board.to_fen())
            temp_board.make_move(move)
            input_data = self.board_to_input(temp_board)
            model_score = self.model.predict(input_data, verbose=0)[0][0]
            evaluation_score = self.evaluate_position(temp_board) / 20
            score = model_score * difficulty + evaluation_score * (1 - difficulty)

            if board.turn == 'w':
                if score > best_score:
                    best_score = score
                    best_move = move
            else:
                if score < best_score:
                    best_score = score
                    best_move = move

        return best_move