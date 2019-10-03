import numpy as np


class StateEncoder:

	def __init__(self):
		pass

	def encode(self, state):
		encoded_state = np.zeros(32)

		for piece in state.board.pieces:
			if piece.captured:
				continue
			if state.whose_turn() == 1:
				if piece.player == state.whose_turn():
					if piece.king:
						encoded_state[piece.position - 1] = 2
					else:
						encoded_state[piece.position - 1] = 1
				else:
					if piece.king:
						encoded_state[piece.position - 1] = -2
					else:
						encoded_state[piece.position - 1] = -1
			else:
				if piece.player == state.whose_turn():
					if piece.king:
						encoded_state[32 - piece.position] = 2
					else:
						encoded_state[32 - piece.position] = 1
				else:
					if piece.king:
						encoded_state[32 - piece.position] = -2
					else:
						encoded_state[32 - piece.position] = -1
		return encoded_state
