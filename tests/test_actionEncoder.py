from unittest import TestCase
from unittest.mock import MagicMock

import numpy as np

from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver


class TestActionEncoder(TestCase):

	def test_encode(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.encode([10, 17])

		expected_action = np.zeros((1, 32 * 4 * 7))
		expected_action[0, 136] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_encode_2(self):
		encoder = ActionEncoder(DirectionResolver())
		actual_action = encoder.encode([5, 32])

		expected_action = np.zeros((1, 32 * 4 * 7))
		expected_action[0, 739] = 1
		self.assertTrue(np.array_equal(actual_action.toarray(), expected_action))

	def test_decode_1(self):
		encoder = ActionEncoder(DirectionResolver())
		encoded_action = np.zeros((1, 32 * 4 * 7))
		encoded_action[0, 136] = 1

		action, direction, distance = encoder.decode(encoded_action)
		self.assertEqual(action, 10)
		self.assertEqual(direction, 1)
		self.assertEqual(distance, 2)

	def test_decode_2(self):
		encoder = ActionEncoder(DirectionResolver())
		encoded_action = np.zeros((1, 32 * 4 * 7))
		encoded_action[0, 739] = 1

		action, direction, distance = encoder.decode(encoded_action)
		self.assertEqual(action, 5)
		self.assertEqual(direction, 4)
		self.assertEqual(distance, 6)