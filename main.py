from flask import Flask, render_template, request, jsonify
from chess_ai import ChessAI, ChessBoard
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
    """Get AI move for AI vs AI mode"""
    try:
        fen = request.json['fen']
        difficulty = float(request.json.get('difficulty', 0.8))

        print(f"AI move requested for FEN: {fen}")

        # Create temporary board from FEN
        temp_board = ChessBoard(fen)

        # Validate that the board loaded correctly
        if not temp_board:
            print("Failed to create board from FEN")
            return jsonify({'error': 'Invalid FEN'}), 400

        # Get legal moves to verify board state
        legal_moves = temp_board.get_legal_moves_safe()
        print(f"Legal moves available: {len(legal_moves)}")

        if not legal_moves:
            print("No legal moves available - game should be over")
            return jsonify({'error': 'No legal moves available'}), 400

        # Get AI move with timeout protection
        move = None
        try:
            move = ai.get_best_move(temp_board, difficulty)
            print(f"AI generated move: {move}")
        except Exception as ai_error:
            print(f"AI move generation failed: {ai_error}")
            # Fallback to random legal move
            move = random.choice(legal_moves)
            print(f"Using random fallback move: {move}")

        if move and move in legal_moves:
            # Validate move is actually legal
            try:
                # Make the move on temporary board to get evaluation
                temp_board.make_move(move)
                evaluation = ai.evaluate_position(temp_board)

                response = {
                    'move': move,
                    'evaluation': evaluation,
                    'fen': temp_board.to_fen()
                }
                print(f"Returning valid move: {move}")
                return jsonify(response)
            except Exception as move_error:
                print(f"Failed to make move {move}: {move_error}")
                # Try a different random move
                remaining_moves = [m for m in legal_moves if m != move]
                if remaining_moves:
                    fallback_move = random.choice(remaining_moves)
                    print(f"Trying fallback move: {fallback_move}")
                    try:
                        temp_board = ChessBoard(fen)  # Reset board
                        temp_board.make_move(fallback_move)
                        evaluation = ai.evaluate_position(temp_board)
                        response = {
                            'move': fallback_move,
                            'evaluation': evaluation,
                            'fen': temp_board.to_fen()
                        }
                        return jsonify(response)
                    except:
                        pass

        print(f"All move attempts failed. Move: {move}, Legal moves: {legal_moves[:5]}")
        return jsonify({'error': f'Generated invalid move: {move}'}), 400

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