from unittest import TestCase

import numpy as np

from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver


class TestActionEncoder(TestCase):

	def test_convert_to_one_hot_3(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([1, 5])

		expected_action = np.zeros((1, 32 * 4))
		expected_action[0, 0] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_to_one_hot_1(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([10, 17])

		expected_action = np.zeros((1, 32 * 4))
		expected_action[0, 9] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_to_one_hot_2(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([5, 32])

		expected_action = np.zeros((1, 32 * 4))
		expected_action[0, 100] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_moves_to_action_ids(self):
		encoder = ActionEncoder(DirectionResolver())
		values = encoder.convert_moves_to_action_ids([[10, 17], [10, 15]])
		self.assertCountEqual(values, [9, 105])

	def test_convert_position_direction_distance_to_move_2(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_direction_and_distance_to_move(10, 1, 2)
		self.assertEqual(move, [10, 17])

	def test_convert_action_id_to_position_and_direction(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_position_and_direction(9)
		self.assertEqual(move, (10, 1))

	def test_convert_action_id_to_true_position_and_direction(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_true_position_and_direction(9, 1)
		self.assertEqual(move, (10, 1))

	def test_convert_action_id_to_true_position_and_direction_2(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_true_position_and_direction(9, 2)
		self.assertEqual(move, (23, 3))

	def test_convert_position_direction_distance_to_move_1(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_direction_and_distance_to_move(10, 4, 1)
		self.assertEqual(move, [10, 15])

	def test_convert_action_id_to_position_and_direction_2(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_position_and_direction(105)
		self.assertEqual(move, (10, 4))

	def test_convert_action_id_to_true_position_and_direction_3(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_true_position_and_direction(105, 1)
		self.assertEqual(move, (10, 4))

	def test_convert_action_id_to_true_position_and_direction_4(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_true_position_and_direction(105, 2)
		self.assertEqual(move, (23, 2))
