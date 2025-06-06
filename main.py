from flask import Flask, render_template, request, jsonify
from chess_ai import ChessAI, ChessBoard

app = Flask(__name__)
board = ChessBoard()
ai = ChessAI()

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

@app.route('/train_ai', methods=['POST'])
def train_ai():
    print("Received train_ai request")
    num_games = int(request.json.get('num_games', 100))
    ai.train_self_play(num_games=num_games)
    response = {'message': f'AI training completed ({num_games} games)'}
    print("Sending response:", response)
    return jsonify(response)

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