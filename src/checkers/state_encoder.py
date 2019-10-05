import numpy as np

from src.checkers.direction_resolver import DirectionResolver


class StateEncoder:

	def __init__(self):
		self.direction_resolver = DirectionResolver()

	def encode(self, state):
		encoded_state = np.zeros((2, 4, 8))

		for piece in state.board.pieces:
			if piece.captured:
				continue
			if state.whose_turn() == 1:
				x, y = self.direction_resolver.get_compressed_coordinates(piece.position)
				if piece.player == state.whose_turn():
					if piece.king:
						encoded_state[0, y, x] = 2
					else:
						encoded_state[0, y, x] = 1
				else:
					if piece.king:
						encoded_state[1, y, x] = 2
					else:
						encoded_state[1, y, x] = 1
			else:
				x, y = self.direction_resolver.get_compressed_coordinates(33 - piece.position)
				if piece.player == state.whose_turn():
					if piece.king:
						encoded_state[0, y, x] = 2
					else:
						encoded_state[0, y, x] = 1
				else:
					if piece.king:
						encoded_state[1, y, x] = 2
					else:
						encoded_state[1, y, x] = 1
		return encoded_state
