from unittest import TestCase

import numpy as np

from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver


class TestActionEncoder(TestCase):

	def test_convert_to_one_hot_3(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([1, 5])

		expected_action = np.zeros((1, 32 * 4 * 7))
		expected_action[0, 0] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_to_one_hot_1(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([10, 17])

		expected_action = np.zeros((1, 32 * 4 * 7))
		expected_action[0, 137] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_to_one_hot_2(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.convert_action_to_one_hot([5, 32])

		expected_action = np.zeros((1, 32 * 4 * 7))
		expected_action[0, 740] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_convert_one_hot_to_directed_action_1(self):
		encoder = ActionEncoder(DirectionResolver())
		encoded_action = np.zeros((1, 32 * 4 * 7))
		encoded_action[0, 137] = 1

		action, direction, distance = encoder.convert_one_hot_to_directed_action(encoded_action)
		self.assertEqual(action, 10)
		self.assertEqual(direction, 1)
		self.assertEqual(distance, 2)

	def test_convert_one_hot_to_directed_action_2(self):
		encoder = ActionEncoder(DirectionResolver())
		encoded_action = np.zeros((1, 32 * 4 * 7))
		encoded_action[0, 740] = 1

		action, direction, distance = encoder.convert_one_hot_to_directed_action(encoded_action)
		self.assertEqual(action, 5)
		self.assertEqual(direction, 4)
		self.assertEqual(distance, 6)

	def test_convert_actions_to_values(self):
		encoder = ActionEncoder(DirectionResolver())
		values = encoder.convert_actions_to_values([[10, 17], [10, 15]])
		self.assertCountEqual(values, [137, 105])

	def test_convert_action_id_to_move_1(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_move(137)
		self.assertEqual(move, [10, 17])

	def test_convert_action_id_to_move_2(self):
		encoder = ActionEncoder(DirectionResolver())
		move = encoder.convert_action_id_to_move(105)
		self.assertEqual(move, [10, 15])
