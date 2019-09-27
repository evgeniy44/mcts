from unittest import TestCase
from unittest.mock import MagicMock

from src.agent import Agent
from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.game import Game
from src.edge import Edge
from src.node import Node


class TestAgent(TestCase):

	def test_act_tau_0(self):
		config = {
			'ALPHA': 0.8,
			'CPUCT': 1,
			'EPSILON': 0.2,
			'ACTION_SIZE': 32 * 4 * 7,
			'MCTS_SIMULATIONS': 3
		}
		agent = Agent(model=None, action_encoder=ActionEncoder(DirectionResolver()), config=config)
		game_root = Game()
		root_node = Node(game_root)

		child1 = Node(game_root.move(game_root.get_possible_moves()[0]))
		edge1 = Edge(root_node, child1, 0.33, [9, 13])
		edge1.stats['N'] = 10
		edge1.stats['Q'] = 0.2

		root_node.edges.append(edge1)

		child2 = Node(game_root.move(game_root.get_possible_moves()[1]))
		edge2 = Edge(root_node, child2, 0.5, [9, 14])
		edge2.stats['N'] = 20
		edge2.stats['Q'] = 0.5
		root_node.edges.append(edge2)

		child3 = Node(game_root.move(game_root.get_possible_moves()[2]))
		edge3 = Edge(root_node, child3, 0.17, [10, 14])
		edge3.stats['N'] = 15
		edge3.stats['Q'] = 0.3
		root_node.edges.append(edge3)

		agent.prepare_mcts_for_next_action = MagicMock()
		mcts = MagicMock()
		mcts.root = root_node
		mcts.evaluate_leaf.return_value = 0.7
		agent.mcts = mcts
		mcts.move_to_leaf.return_value = (root_node, 0.5, False, [])

		action, pi, value = agent.act(game_root, tau=0)

		self.assertEqual(action, [9, 14])
		self.assertEqual(value, 0.5)
		self.assertEqual(pi[8], 10/(10 + 20 + 15))
		self.assertEqual(pi[9], 15/(10 + 20 + 15))
		self.assertEqual(pi[8 + 3*32], 20/(10 + 20 + 15))
