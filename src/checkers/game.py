from .board import Board
import numpy as np

PLAYER_TURN_POSITION = 32


class Game:

	def __init__(self, board=Board(), moves=None, moves_since_last_capture=0):
		if moves is None:
			moves = []
		self.board = board
		self.moves = moves
		self.consecutive_noncapture_move_limit = 40
		self.moves_since_last_capture = moves_since_last_capture

	def move(self, move):
		if move not in self.get_possible_moves():
			raise ValueError('The provided move is not possible')

		board = self.board.create_new_board_from_move(move)
		moves = self.moves[:]
		moves.append(move)
		moves_since_last_capture = 0 if board.previous_move_was_capture else self.moves_since_last_capture + 1

		return Game(board=board, moves=moves, moves_since_last_capture=moves_since_last_capture)

	def move_limit_reached(self):
		return self.moves_since_last_capture >= self.consecutive_noncapture_move_limit

	def is_over(self):
		return self.move_limit_reached() or not self.get_possible_moves()

	def get_winner(self):
		if not self.board.count_movable_player_pieces(1):
			return 2
		elif not self.board.count_movable_player_pieces(2):
			return 1
		else:
			return None

	def get_possible_moves(self):
		return self.board.get_possible_moves()

	def get_possible_moves_from_current_player_perspective(self):
		if self.whose_turn() == 1:
			return self.board.get_possible_moves()
		else:
			return list(map(self.convert_move_to_black_perspective, self.board.get_possible_moves()))

	def convert_move_to_black_perspective(self, move):
		return list(map(lambda x: 33 - x, move))

	def whose_turn(self):
		return self.board.player_turn

	def id(self):
		id = np.zeros(33)
		id[PLAYER_TURN_POSITION] = self.whose_turn()
		for piece in self.board.pieces:
			id[piece.position - 1] = piece.player
		return id

	def render(self):
		rows = [
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0),
			np.full(shape=8, fill_value=0)
		]
		for piece in self.board.pieces:
			if piece.position is None:
				continue
			y = (piece.position - 1) // 4
			x = (3 - ((piece.position - 1) % 4)) * 2
			if y % 2 == 1:
				x = x + 1
			rows[y][x] = piece.player

		print("Current game state: ")
		for i in range(7, -1, -1):
			print(rows[i])
