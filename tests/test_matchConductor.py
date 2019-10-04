import random
from unittest import TestCase
from unittest.mock import MagicMock, PropertyMock

import numpy as np

from src.match_conductor import MatchConductor
from src.memory import Memory


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
		next_state.get_winner_for_learning.return_value = 1

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1), 1)

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10)

		self.assertEqual(scores['player1'], 1)
		self.assertEqual(scores['player2'], 0)
		self.assertEqual(scores['drawn'], 0)

	def test_play_matches_with_black_win(self):
		np.random.seed(1)
		random.seed(5)

		player1 = MagicMock()
		player2 = MagicMock()
		initial_state = MagicMock()
		next_state = MagicMock()
		next_next_state = MagicMock()

		conductor = MatchConductor(initial_state)

		initial_state.move.return_value = next_state
		initial_state.whose_turn.return_value = 1

		next_state.is_over.return_value = False
		next_state.move.return_value = next_next_state
		next_state.get_winner.return_value = 1
		next_state.whose_turn.return_value = 2
		next_state.opposite_turn.return_value = 1
		next_state.get_winner_for_learning.return_value = 1

		next_next_state.is_over.return_value = True
		next_next_state.get_winner.return_value = 1
		next_next_state.whose_turn.return_value = 1
		next_next_state.opposite_turn.return_value = 2
		next_next_state.get_winner_for_learning.return_value = 1

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1), 1)
		player2.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.2), 1)

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10)

		self.assertEqual(scores['player1'], 0)
		self.assertEqual(scores['player2'], 1)
		self.assertEqual(scores['drawn'], 0)

	def test_play_matches_with_black_win_and_memory(self):
		np.random.seed(1)
		random.seed(5)

		player1 = MagicMock()
		player2 = MagicMock()
		initial_state = MagicMock()
		next_state = MagicMock()
		next_next_state = MagicMock()

		conductor = MatchConductor(initial_state)

		initial_state.move.return_value = next_state
		initial_state.whose_turn.return_value = 1

		next_state.is_over.return_value = False
		next_state.move.return_value = next_next_state
		next_state.get_winner.return_value = 1
		next_state.whose_turn.return_value = 2
		next_state.opposite_turn.return_value = 1
		next_state.get_winner_for_learning.return_value = 1

		next_next_state.is_over.return_value = True
		next_next_state.get_winner.return_value = 1
		next_next_state.whose_turn.return_value = 1
		next_next_state.opposite_turn.return_value = 2
		next_next_state.get_winner_for_learning.return_value = 1

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1), 1)
		player2.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.2), 1)

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		memory = Memory(100)
		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10, memory=memory)

		self.assertEqual(scores['player1'], 0)
		self.assertEqual(scores['player2'], 1)
		self.assertEqual(scores['drawn'], 0)

		self.assertEqual(memory.ltmemory[0]['state'], initial_state)
		self.assertTrue(np.array_equal(memory.ltmemory[0]['AV'], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1)))
		self.assertEqual(memory.ltmemory[0]['value'], -1)

		self.assertEqual(memory.ltmemory[1]['state'], next_state)
		self.assertTrue(np.array_equal(memory.ltmemory[1]['AV'], np.full(shape=(1, 32 * 4 * 7), fill_value=0.2)))
		self.assertEqual(memory.ltmemory[1]['value'], 1)

	def test_play_matches_with_memory(self):
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
		next_state.get_winner_for_learning.return_value = 1
		next_state.opposite_turn.return_value = 1

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1), 1)

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		memory = Memory(100)
		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10, memory=memory)

		self.assertEqual(scores['player1'], 1)
		self.assertEqual(scores['player2'], 0)
		self.assertEqual(scores['drawn'], 0)

		self.assertEqual(memory.ltmemory[0]['state'], initial_state)
		self.assertTrue(np.array_equal(memory.ltmemory[0]['AV'], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1)))
		self.assertEqual(memory.ltmemory[0]['value'], 1)

	def test_play_matches_with__draw_and_memory(self):
		np.random.seed(1)
		random.seed(5)

		player1 = MagicMock()
		player2 = MagicMock()
		initial_state = MagicMock()
		next_state = MagicMock()
		next_next_state = MagicMock()

		conductor = MatchConductor(initial_state)

		initial_state.move.return_value = next_state
		initial_state.whose_turn.return_value = 1

		next_state.is_over.return_value = False
		next_state.move.return_value = next_next_state
		next_state.get_winner.return_value = 1
		next_state.whose_turn.return_value = 2
		next_state.opposite_turn.return_value = 1
		next_state.get_winner_for_learning.return_value = 1

		next_next_state.is_over.return_value = True
		next_next_state.get_winner.return_value = 1
		next_next_state.whose_turn.return_value = 1
		next_next_state.opposite_turn.return_value = 2
		next_next_state.get_winner_for_learning.return_value = 0

		player1.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1), 1)
		player2.act.return_value = ([9, 14], np.full(shape=(1, 32 * 4 * 7), fill_value=0.2), 1)

		type(player1).name = PropertyMock(return_value='player1')
		type(player2).name = PropertyMock(return_value='player2')

		player1.name = PropertyMock(return_value='player1')
		player2.name = PropertyMock(return_value='player2')

		memory = Memory(100)
		scores, memory = conductor.play_matches(player1, player2, episodes_count=1, turns_until_tau0=10, memory=memory)

		self.assertEqual(scores['player1'], 0)
		self.assertEqual(scores['player2'], 0)
		self.assertEqual(scores['drawn'], 1)

		self.assertEqual(memory.ltmemory[0]['state'], initial_state)
		self.assertTrue(np.array_equal(memory.ltmemory[0]['AV'], np.full(shape=(1, 32 * 4 * 7), fill_value=0.1)))
		self.assertEqual(memory.ltmemory[0]['value'], 0)

		self.assertEqual(memory.ltmemory[1]['state'], next_state)
		self.assertTrue(np.array_equal(memory.ltmemory[1]['AV'], np.full(shape=(1, 32 * 4 * 7), fill_value=0.2)))
		self.assertEqual(memory.ltmemory[1]['value'], 0)