from unittest import TestCase

import numpy as np

from src.checkers.game import Game


class GameTest(TestCase):

	def test_id_white(self):
		game = Game()
		expected_id = np.zeros(33)
		expected_id[32] = 1
		expected_id[:12] = np.ones(12)
		expected_id[20:32] = np.full(shape=12, fill_value=2)
		self.assertTrue(np.array_equal(game.id(), expected_id))

	def test_id_black(self):
		game = Game()
		game = game.move(game.get_possible_moves()[0])
		expected_id = np.zeros(33)
		expected_id[32] = 2
		expected_id[:13] = np.ones(13)
		expected_id[8] = 0
		expected_id[20:32] = np.full(shape=12, fill_value=2)
		self.assertTrue(np.array_equal(game.id(), expected_id))

	def test_moves_from_current_player_perspective_white(self):
		game = Game()
		moves = game.get_possible_moves_from_current_player_perspective()
		self.assertCountEqual(moves, [[9,13], [9,14], [10, 14], [10,15], [11, 15], [11, 16], [12, 16]])

	def test_moves_from_current_player_perspective_black(self):
		game = Game()
		game = game.move([9, 13])
		moves = game.get_possible_moves_from_current_player_perspective()
		self.assertCountEqual(moves, [[9,13], [9,14], [10, 14], [10,15], [11, 15], [11, 16], [12, 16]])

	def test_move_with_jumps(self):
		game = Game()

		game = game.move_with_additional_jumps([12, 16])
		self.assertEqual(game.moves, [[12, 16]])

		game = game.move_with_additional_jumps([23, 18])
		self.assertEqual(game.moves, [[12, 16], [23, 18]])

		game = game.move_with_additional_jumps([8, 12])
		self.assertEqual(game.moves, [[12, 16], [23, 18], [8, 12]])

		game = game.move_with_additional_jumps([27, 23])
		self.assertEqual(game.moves, [[12, 16], [23, 18], [8, 12], [27, 23]])

		game = game.move_with_additional_jumps([4, 8])
		self.assertEqual(game.moves, [[12, 16], [23, 18], [8, 12], [27, 23], [4, 8]])

		game = game.move_with_additional_jumps([18, 14])
		self.assertEqual(game.moves, [[12, 16], [23, 18], [8, 12], [27, 23], [4, 8], [18, 14]])

		game = game.move_with_additional_jumps([9, 18])
		self.assertEqual(game.moves, [[12, 16], [23, 18], [8, 12], [27, 23], [4, 8], [18, 14], [9, 18], [18, 27]])
