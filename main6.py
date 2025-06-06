import chess
import chess.engine
import chess.pgn
import chess.polyglot
import random
import PySimpleGUI as sg
import threading
import time
import io
import json

# ... (previous code for piece-square tables and evaluation function remains the same)

class ChessGame:
    def __init__(self, engine_path):
        self.board = chess.Board()
        self.engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        self.window = None
        self.move_history = []
        self.current_move = -1
        self.ai_depth = 4
        self.ai_skill_level = 20
        self.white_time = 600  # 10 minutes
        self.black_time = 600  # 10 minutes
        self.clock_running = False
        self.last_move_time = None

    def create_board_image(self):
        # ... (previous code for creating board image remains the same)

    def create_window(self):
        board_image = self.create_board_image()
        move_history = sg.Listbox(values=[], size=(20, 20), key='-MOVE_HISTORY-', enable_events=True)
        
        options_column = [
            [sg.Text('AI Depth:'), sg.Slider(range=(1, 10), default_value=4, orientation='h', key='-AI_DEPTH-')],
            [sg.Text('AI Skill:'), sg.Slider(range=(0, 20), default_value=20, orientation='h', key='-AI_SKILL-')],
            [sg.Button('New Game'), sg.Button('Save Game'), sg.Button('Load Game'), sg.Button('Exit')]
        ]
        
        clock_column = [
            [sg.Text('White: 10:00', key='-WHITE_CLOCK-')],
            [sg.Text('Black: 10:00', key='-BLACK_CLOCK-')]
        ]
        
        layout = [
            [sg.Column(board_image), sg.Column([[move_history], [sg.Button('<<'), sg.Button('<'), sg.Button('>'), sg.Button('>>')]])],
            [sg.Column(options_column), sg.Column(clock_column)],
            [sg.Text('', size=(40, 1), key='-STATUS-')]
        ]
        self.window = sg.Window('Chess Game', layout, finalize=True)
        
        # Start clock update thread
        threading.Thread(target=self.update_clock, daemon=True).start()

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
        
        # Update move history
        self.window['-MOVE_HISTORY-'].update(values=self.move_history[:self.current_move+1])

    def update_clock(self):
        while True:
            if self.clock_running:
                current_time = time.time()
                if self.last_move_time is not None:
                    elapsed = current_time - self.last_move_time
                    if self.board.turn == chess.WHITE:
                        self.white_time -= elapsed
                    else:
                        self.black_time -= elapsed
                self.last_move_time = current_time
                
                self.window['-WHITE_CLOCK-'].update(f'White: {int(self.white_time // 60):02d}:{int(self.white_time % 60):02d}')
                self.window['-BLACK_CLOCK-'].update(f'Black: {int(self.black_time // 60):02d}:{int(self.black_time % 60):02d}')
            
            time.sleep(0.1)

    def get_best_move(self, board, depth):
        # ... (previous code for get_best_move remains the same)

    def make_move(self, move):
        self.board.push(move)
        self.move_history = self.move_history[:self.current_move+1]
        self.move_history.append(move.uci())
        self.current_move += 1
        self.update_board()
        
        # Switch clock
        if self.last_move_time is not None:
            elapsed = time.time() - self.last_move_time
            if self.board.turn == chess.BLACK:
                self.white_time -= elapsed
            else:
                self.black_time -= elapsed
        self.last_move_time = time.time()
        
        if self.board.is_game_over():
            self.clock_running = False
            self.window['-STATUS-'].update('Game Over')
        else:
            self.window['-STATUS-'].update('Your turn' if self.board.turn == chess.WHITE else 'AI is thinking...')

    def ai_move(self):
        ai_move = self.get_best_move(self.board, self.ai_depth)
        self.make_move(ai_move)

    def save_game(self):
        game = chess.pgn.Game.from_board(self.board)
        game.headers["Event"] = "Chess AI Game"
        game.headers["Site"] = "PySimpleGUI Chess"
        game.headers["Date"] = time.strftime("%Y.%m.%d")
        
        exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=True)
        pgn_string = game.accept(exporter)
        
        filename = sg.popup_get_file('Save game as PGN', save_as=True, file_types=(("PGN Files", "*.pgn"),))
        if filename:
            with open(filename, 'w') as f:
                f.write(pgn_string)
            sg.popup('Game saved successfully!')

    def load_game(self):
        filename = sg.popup_get_file('Load PGN file', file_types=(("PGN Files", "*.pgn"),))
        if filename:
            with open(filename) as f:
                game = chess.pgn.read_game(f)
                if game:
                    self.board = game.board()
                    for move in game.mainline_moves():
                        self.board.push(move)
                    self.move_history = [move.uci() for move in game.mainline_moves()]
                    self.current_move = len(self.move_history) - 1
                    self.update_board()
                    sg.popup('Game loaded successfully!')
                else:
                    sg.popup('Invalid PGN file!')

    def play(self):
        self.create_window()
        self.clock_running = True
        self.last_move_time = time.time()

        while True:
            event, values = self.window.read(timeout=100)
            if event in (sg.WINDOW_CLOSED, 'Exit'):
                break
            elif event == 'New Game':
                self.board.reset()
                self.move_history = []
                self.current_move = -1
                self.white_time = 600
                self.black_time = 600
                self.clock_running = True
                self.last_move_time = time.time()
                self.update_board()
                self.window['-STATUS-'].update('Your turn (White)')
            elif event == 'Save Game':
                self.save_game()
            elif event == 'Load Game':
                self.load_game()
            elif event == '-MOVE_HISTORY-':
                if values['-MOVE_HISTORY-']:
                    move_index = self.move_history.index(values['-MOVE_HISTORY-'][0])
                    self.board = chess.Board()
                    for move in self.move_history[:move_index+1]:
                        self.board.push(chess.Move.from_uci(move))
                    self.current_move = move_index
                    self.update_board()
            elif event in ('<<', '<', '>', '>>'):
                if event == '<<':
                    self.current_move = -1
                elif event == '<' and self.current_move > -1:
                    self.current_move -= 1
                elif event == '>' and self.current_move < len(self.move_history) - 1:
                    self.current_move += 1
                elif event == '>>':
                    self.current_move = len(self.move_history) - 1
                
                self.board = chess.Board()
                for move in self.move_history[:self.current_move+1]:
                    self.board.push(chess.Move.from_uci(move))
                self.update_board()
            elif isinstance(event, tuple):  # Square clicked
                i, j = event
                square = chess.square(j, 7-i)
                move = chess.Move.from_uci(f"{chess.SQUARE_NAMES[square]}{chess.SQUARE_NAMES[square]}")
                if move in self.board.legal_moves:
                    self.make_move(move)
                    if not self.board.is_game_over():
                        self.window['-STATUS-'].update('AI is thinking...')
                        self.window.refresh()
                        self.ai_depth = int(values['-AI_DEPTH-'])
                        self.ai_skill_level = int(values['-AI_SKILL-'])
                        self.engine.configure({"Skill Level": self.ai_skill_level})
                        threading.Thread(target=self.ai_move, daemon=True).start()
                else:
                    self.window['-STATUS-'].update('Invalid move. Try again.')
            
            # Check for game over conditions
            if self.board.is_game_over():
                self.clock_running = False
                result = self.board.result()
                if result == '1-0':
                    self.window['-STATUS-'].update('Game Over. White wins!')
                elif result == '0-1':
                    self.window['-STATUS-'].update('Game Over. Black wins!')
                else:
                    self.window['-STATUS-'].update('Game Over. It\'s a draw!')
            
            # Check for time out
            if self.white_time <= 0:
                self.clock_running = False
                self.window['-STATUS-'].update('Game Over. Black wins on time!')
            elif self.black_time <= 0:
                self.clock_running = False
                self.window['-STATUS-'].update('Game Over. White wins on time!')

        self.window.close()
        self.engine.quit()

if __name__ == "__main__":
    stockfish_path = "/path/to/stockfish"  # Update this with your Stockfish engine path
    game = ChessGame(stockfish_path)
    game.play()