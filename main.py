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
        ai_move = ai.get_best_move(board, difficulty)
        if ai_move:
            board.make_move(ai_move)
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
                legal_moves = board.get_legal_moves()

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
    legal_moves = board.get_legal_moves()
    return jsonify({'legal_moves': legal_moves})

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