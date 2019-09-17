from unittest import TestCase

from src.checkers.game import Game
from src.edge import Edge
from src.mcts import MCTS
from src.node import Node
from unittest.mock import MagicMock


class TestMCTS(TestCase):

	def test_move_to_leaf(self):
		game = Game()
		root = Node(game)
		mcts = MCTS(root, {
			'ALPHA': 0.8,
			'CPUCT': 1,
			'EPSILON': 0.2
		})

		puct = MagicMock()
		mcts.puct = puct

		child1 = Node(game.move(game.get_possible_moves()[0]))
		child2 = Node(game.move(game.get_possible_moves()[1]))
		child3 = Node(game.move(game.get_possible_moves()[2]))
		edge1 = Edge(root, child1, 0.33, game.get_possible_moves()[0])
		edge2 = Edge(root, child2, 0.34, game.get_possible_moves()[1])
		edge3 = Edge(root, child3, 0.33, game.get_possible_moves()[2])
		root.edges.append(edge1)
		root.edges.append(edge2)
		root.edges.append(edge3)
		puct.puct.return_value = edge2

		leaf, value, done, breadcrumbs = mcts.move_to_leaf()

		self.assertEquals(leaf, child2)
		self.assertEquals(value, 0)
		self.assertEquals(done, 0)
		self.assertEquals(False, 0)
		self.assertEquals(True, 1)
