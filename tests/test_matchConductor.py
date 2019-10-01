import random
from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from src.match_conductor import MatchConductor


class TestMatchConductor(TestCase):

	def test_play_matches(self):
		np.random.seed(1)
		random.seed(5)

		player1 = MagicMock()
		player2 = MagicMock()
		initial_state = MagicMock()
		next_state = MagicMock(name='next_state')

		conductor = MatchConductor(initial_state)

		initial_state.move.return_value = next_state
		initial_state.whose_turn.return_value = 1
		next_state.is_over.return_value = True
		next_state.get_winner.return_value = 1
		next_state.whose_turn.return_value = 2
		next_state.opposite_turn.return_value = 1

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1))

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10)

		self.assertEqual(scores['player1'], 1)
		self.assertEqual(scores['player2'], 0)
		self.assertEqual(scores['drawn'], 0)
