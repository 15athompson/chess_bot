from flask import Flask, render_template, request, jsonify
from chess_ai import ChessAI, ChessBoard
import chess
import chess.engine
import threading
import time
import numpy as np
import random

app = Flask(__name__)
board = ChessBoard()
ai = ChessAI()

# Training status tracking
training_status = {
    'is_training': False,
    'progress': 0,
    'total_games': 0,
    'current_game': 0,
    'message': 'Ready to train'
}

def select_smart_move(chess_board, legal_moves):
    """Select a smart fallback move when AI fails"""
    if not legal_moves:
        return None

    # Priority 1: Captures
    captures = [move for move in legal_moves if chess_board.is_capture(move)]
    if captures:
        return captures[0]

    # Priority 2: Checks
    checks = []
    for move in legal_moves:
        chess_board.push(move)
        if chess_board.is_check():
            checks.append(move)
        chess_board.pop()
    if checks:
        return checks[0]

    # Priority 3: Center control
    center_squares = [chess.E4, chess.E5, chess.D4, chess.D5]
    center_moves = [move for move in legal_moves if move.to_square in center_squares]
    if center_moves:
        return center_moves[0]

    # Priority 4: Piece development (knights and bishops)
    development_moves = []
    for move in legal_moves:
        piece = chess_board.piece_at(move.from_square)
        if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            # Avoid moving to edge squares
            if move.to_square not in [chess.A1, chess.A8, chess.H1, chess.H8]:
                development_moves.append(move)
    if development_moves:
        return development_moves[0]

    # Priority 5: Pawn moves
    pawn_moves = []
    for move in legal_moves:
        piece = chess_board.piece_at(move.from_square)
        if piece and piece.piece_type == chess.PAWN:
            pawn_moves.append(move)
    if pawn_moves:
        return pawn_moves[0]

    # Fallback: first legal move
    return legal_moves[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/make_move', methods=['POST'])
def make_move():
    print("Received make_move request")
    move = request.json['move']
    difficulty = float(request.json.get('difficulty', 1.0))
    try:
        board.make_move(move)
        print(f"Player move: {move}")

        # Check if game is over after player move
        if board.is_game_over():
            result = board.get_result()
            print(f"Game over after player move! Result: {result}")
            response = {
                'fen': board.to_fen(),
                'ai_move': None,
                'game_over': True,
                'result': result,
                'evaluation': ai.evaluate_position(board)
            }
        else:
            # Get AI move with timeout protection
            try:
                print("Getting AI move...")
                ai_move = ai.get_best_move(board, difficulty)
                if ai_move:
                    print(f"AI move: {ai_move}")
                    board.make_move(ai_move)
                else:
                    print("AI has no legal moves!")
            except Exception as e:
                print(f"AI error: {e}")
                # Emergency fallback - pick first legal move
                legal_moves = board.get_legal_moves_safe()
                if legal_moves:
                    ai_move = legal_moves[0]
                    print(f"Emergency AI move: {ai_move}")
                    board.make_move(ai_move)
                else:
                    ai_move = None

            response = {
                'fen': board.to_fen(),
                'ai_move': ai_move,
                'game_over': board.is_game_over(),
                'result': board.get_result() if board.is_game_over() else None,
                'evaluation': ai.evaluate_position(board)
            }
        print("Sending response:", response)
        return jsonify(response)
    except ValueError as e:
        print("Error in make_move:", str(e))
        return jsonify({'error': 'Invalid move'}), 400

@app.route('/new_game', methods=['POST'])
def new_game():
    print("Received new_game request")
    global board
    board = ChessBoard()
    # Clear move history for new game
    board.move_history = []
    board.board_history = []
    response = {'fen': board.to_fen()}
    print("Sending response:", response)
    return jsonify(response)

def background_training(num_games):
    """Run training in background thread"""
    global training_status
    try:
        training_status['is_training'] = True
        training_status['total_games'] = num_games
        training_status['current_game'] = 0
        training_status['message'] = 'Training in progress...'

        print(f"Starting background training with {num_games} games...")

        # Modified training loop with progress updates
        for game_num in range(num_games):
            training_status['current_game'] = game_num + 1
            training_status['progress'] = int((game_num + 1) / num_games * 100)

            # Train one game
            board = ChessBoard()
            game_history = []
            move_count = 0
            max_moves = 50  # Shorter games for faster training

            while not board.is_game_over() and move_count < max_moves:
                input_data = ai.board_to_input(board)
                legal_moves = board.get_legal_moves_safe()  # Use safe legal moves

                if not legal_moves:
                    break

                # Simple random move selection for training
                move = random.choice(legal_moves)
                game_history.append((input_data, ai.evaluate_position(board)))
                board.make_move(move)
                move_count += 1

            # Train the model on this game
            if game_history:
                reward = 0  # Simplified reward
                for input_data, evaluation in reversed(game_history):
                    target = reward + 0.9 * evaluation / 20
                    ai.model.fit(input_data, np.array([target]), verbose=0)
                    reward = target

        training_status['is_training'] = False
        training_status['message'] = f'Training completed! Trained on {num_games} games.'
        print("Background training completed!")

    except Exception as e:
        training_status['is_training'] = False
        training_status['message'] = f'Training failed: {str(e)}'
        print(f"Training error: {e}")

@app.route('/train_ai', methods=['POST'])
def train_ai():
    print("Received train_ai request")
    global training_status

    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400

    num_games = int(request.json.get('num_games', 50))  # Reduced default

    # Start training in background thread
    training_thread = threading.Thread(target=background_training, args=(num_games,))
    training_thread.daemon = True
    training_thread.start()

    response = {'message': f'Training started with {num_games} games. Check status for progress.'}
    print("Sending response:", response)
    return jsonify(response)

@app.route('/training_status', methods=['GET'])
def get_training_status():
    return jsonify(training_status)

@app.route('/get_legal_moves', methods=['POST'])
def get_legal_moves():
    fen = request.json['fen']
    board = ChessBoard(fen)
    legal_moves = board.get_legal_moves_safe()  # Use safe legal moves
    return jsonify({'legal_moves': legal_moves})

@app.route('/ai_move', methods=['POST'])
def ai_move():
    """Get AI move for AI vs AI mode using python-chess for reliability"""
    try:
        fen = request.json['fen']
        difficulty = float(request.json.get('difficulty', 0.8))

        print(f"AI move requested for FEN: {fen}")

        # Use python-chess for reliable board handling
        chess_board = chess.Board(fen)

        if chess_board.is_game_over():
            print("Game is over according to python-chess")
            return jsonify({'error': 'Game is over'}), 400

        # Get legal moves using python-chess
        legal_moves = list(chess_board.legal_moves)
        print(f"Legal moves available: {len(legal_moves)}")

        if not legal_moves:
            print("No legal moves available")
            return jsonify({'error': 'No legal moves available'}), 400

        # Convert to our custom board for AI evaluation
        custom_board = ChessBoard(fen)

        # Get AI move
        ai_move = ai.get_best_move(custom_board, difficulty)
        print(f"AI generated move: {ai_move}")

        if ai_move:
            # Try multiple approaches to validate and convert the move
            move_found = False
            final_move = None

            # Approach 1: Direct string parsing
            try:
                from_square = chess.parse_square(ai_move[:2])
                to_square = chess.parse_square(ai_move[2:4])

                # Handle promotion
                promotion = None
                if len(ai_move) == 5:
                    promotion_piece = ai_move[4].lower()
                    if promotion_piece == 'q':
                        promotion = chess.QUEEN
                    elif promotion_piece == 'r':
                        promotion = chess.ROOK
                    elif promotion_piece == 'b':
                        promotion = chess.BISHOP
                    elif promotion_piece == 'n':
                        promotion = chess.KNIGHT

                move = chess.Move(from_square, to_square, promotion)

                if move in chess_board.legal_moves:
                    final_move = move
                    move_found = True
                    print(f"Move validated via direct parsing: {ai_move}")

            except Exception as e:
                print(f"Direct parsing failed: {e}")

            # Approach 2: Try to find matching move in legal moves
            if not move_found:
                legal_move_strings = [str(m) for m in legal_moves]

                # Check if the move string matches any legal move
                if ai_move in legal_move_strings:
                    final_move = legal_moves[legal_move_strings.index(ai_move)]
                    move_found = True
                    print(f"Move found in legal moves list: {ai_move}")

                # Try variations (with/without promotion notation)
                elif len(ai_move) == 4:
                    # Try adding common promotions
                    for promo in ['q', 'r', 'b', 'n']:
                        test_move = ai_move + promo
                        if test_move in legal_move_strings:
                            final_move = legal_moves[legal_move_strings.index(test_move)]
                            move_found = True
                            print(f"Move found with promotion: {test_move}")
                            break

            # Approach 3: Fallback to best legal move if AI move is invalid
            used_fallback = False
            if not move_found:
                print(f"AI move {ai_move} not found in legal moves: {legal_move_strings[:10]}")
                # Select a good fallback move
                final_move = select_smart_move(chess_board, legal_moves)
                move_found = True
                used_fallback = True
                print(f"Using smart fallback move: {final_move}")

            if move_found and final_move:
                # Make the move
                chess_board.push(final_move)

                # Get evaluation using our custom board
                try:
                    custom_board.make_move(str(final_move))
                    evaluation = ai.evaluate_position(custom_board)
                except:
                    evaluation = 0  # Fallback evaluation

                response = {
                    'move': str(final_move),
                    'evaluation': evaluation,
                    'fen': chess_board.fen(),
                    'fallback_used': used_fallback
                }
                print(f"Returning move: {final_move} (fallback: {used_fallback})")
                return jsonify(response)
        else:
            print("AI returned no move")
            return jsonify({'error': 'AI failed to generate move'}), 400

    except Exception as e:
        print(f"Error in ai_move: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'AI move failed: {str(e)}'}), 500

@app.route('/undo_move', methods=['POST'])
def undo_move():
    print("Received undo_move request")
    global board

    # Try to undo twice (player move and AI move)
    undone_moves = 0
    if board.can_undo():
        board.undo_move()
        undone_moves += 1
    if board.can_undo():
        board.undo_move()
        undone_moves += 1

    if undone_moves > 0:
        response = {
            'fen': board.to_fen(),
            'undone_moves': undone_moves,
            'evaluation': ai.evaluate_position(board)
        }
        print("Sending response:", response)
        return jsonify(response)
    else:
        print("No moves to undo")
        return jsonify({'error': 'No moves to undo'}), 400

if __name__ == '__main__':
    app.run(debug=True)