from unittest import TestCase

import numpy as np

from src.checkers.game import Game
from src.checkers.state_encoder import StateEncoder


class TestStateEncoder(TestCase):

	def test_encode_white(self):
		game = Game()
		encoder = StateEncoder()

		encoded_state = encoder.encode(game)
		expected = np.array([[
			[
				[1, 1, 1, 1, 1, 1, 1, 1],
				[1, 0, 1, 0, 1, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0]
			],
			[
				[0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0],
				[0, 1, 0, 1, 0, 1, 0, 1],
				[1, 1, 1, 1, 1, 1, 1, 1]
			]
		]], dtype=np.float64)
		self.assertTrue(np.array_equal(encoded_state, expected))

	def test_encode_black(self):
		game = Game()
		game = game.move(game.get_possible_moves()[0])
		encoder = StateEncoder()

		encoded_state = encoder.encode(game)
		expected = np.array([[
			[
				[1, 1, 1, 1, 1, 1, 1, 1],
				[1, 0, 1, 0, 1, 0, 1, 0],
				[0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0]
			],
			[
				[0, 0, 0, 0, 0, 0, 0, 0],
				[0, 0, 0, 0, 0, 0, 0, 0],
				[1, 0, 0, 1, 0, 1, 0, 1],
				[1, 1, 1, 1, 1, 1, 1, 1]
			]
		]], dtype=np.float64)
		self.assertTrue(np.array_equal(encoded_state, expected))
