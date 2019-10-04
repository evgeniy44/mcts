from unittest import TestCase

import numpy as np

from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.game import Game
from src.checkers.state_encoder import StateEncoder
from src.edge import Edge
from src.mcts import MCTS
from src.model import Residual_CNN
from src.node import Node
from unittest.mock import MagicMock


class TestMCTS(TestCase):

    def test_move_to_leaf(self):
        game = Game()
        root = Node(game)
        action_encoder = ActionEncoder(DirectionResolver())
        mcts = MCTS(root, config={
            'ALPHA': 0.8,
            'CPUCT': 1,
            'EPSILON': 0.2
        }, model=None, state_encoder=None, action_encoder=action_encoder)

        puct = MagicMock()
        mcts.puct = puct

        child1 = Node(game.move(game.get_possible_moves()[0]))
        child2 = Node(game.move(game.get_possible_moves()[1]))
        child3 = Node(game.move(game.get_possible_moves()[2]))
        edge1 = Edge(root, child1, 0.33, action_encoder.convert_move_to_action_id(game.get_possible_moves()[0]))
        edge2 = Edge(root, child2, 0.34, action_encoder.convert_move_to_action_id(game.get_possible_moves()[1]))
        edge3 = Edge(root, child3, 0.33, action_encoder.convert_move_to_action_id(game.get_possible_moves()[2]))
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

    def test_backfill(self):
        game_root = Game()
        root = Node(game_root)
        action_encoder = ActionEncoder(DirectionResolver())
        position1 = game_root.move(game_root.get_possible_moves()[0])
        child1 = Node(position1)
        edge1 = Edge(root, child1, 0.3, action_encoder.convert_move_to_action_id(game_root.get_possible_moves()[0]))

        position2 = position1.move(position1.get_possible_moves()[0])
        child2 = Node(position2)
        edge2 = Edge(child1, child2, 0.2, action_encoder.convert_move_to_action_id(game_root.get_possible_moves()[0]))
        edge2.stats['N'] = 4
        edge2.stats['W'] = 1

        mcts = MCTS(root, config={
            'ALPHA': 0.8,
            'CPUCT': 1,
            'EPSILON': 0.2
        }, model=None, state_encoder=None, action_encoder=action_encoder)

        mcts.backfill(child2, -1, [edge2, edge1])

        self.assertEquals(edge2.stats['N'], 5)
        self.assertEquals(edge2.stats['W'], 2)
        self.assertEquals(edge2.stats['Q'], 2 / 5)

        self.assertEquals(edge1.stats['N'], 1)
        self.assertEquals(edge1.stats['W'], -1)
        self.assertEquals(edge1.stats['Q'], -1)

    def test_predict(self):
        game_root = Game()
        root = Node(game_root)
        model = MagicMock()

        prediction = [np.array([[0.25]]), np.reshape(np.arange(0.001, 0.897, step=0.001), newshape=(1, 896))]
        model.predict.return_value = prediction

        action_encoder = ActionEncoder(DirectionResolver())
        mcts = MCTS(root, config={
            'ALPHA': 0.8,
            'CPUCT': 1,
            'EPSILON': 0.2
        }, model=model, state_encoder=StateEncoder(), action_encoder=action_encoder)

        value, probs, allowed_actions = mcts.predict_state_value(game_root)

        self.assertEqual(value, 0.25)
        self.assertCountEqual(allowed_actions, action_encoder.convert_actions_to_values(
            game_root.get_possible_moves_from_current_player_perspective()))
        for idx, prob in enumerate(probs):
            if idx in allowed_actions:
                self.assertTrue(prob > 0.01)
            else:
                self.assertTrue(prob < np.exp(-40))

    def test_evaluate_leaf(self):
        game_root = Game()
        root = Node(game_root)
        model = MagicMock()

        prediction = [np.array([[0.25]]), np.reshape(np.arange(0.001, 0.897, step=0.001), newshape=(1, 896))]
        model.predict.return_value = prediction

        action_encoder = ActionEncoder(DirectionResolver())
        mcts = MCTS(root, config={
            'ALPHA': 0.8,
            'CPUCT': 1,
            'EPSILON': 0.2
        }, model=model, state_encoder=StateEncoder(), action_encoder=action_encoder)
        _, probs, _ = mcts.predict_state_value(game_root)
        value = mcts.evaluate_leaf(root)
        self.assertEqual(value, 0.25)
        self.assertEqual(len(root.edges), 7)
        self.assertEqual(root.edges[0].action, 8)
        self.assertEqual(root.edges[0].stats['P'], probs[8])

        self.assertEqual(root.edges[1].action, 104)
        self.assertEqual(root.edges[1].stats['P'], probs[104])

    def test_integration(self):
        HIDDEN_CNN_LAYERS = [
            {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
            , {'filters': 75, 'kernel_size': (4, 4)}
        ]
        model = Residual_CNN(0.0001, 0.1, (2, 4, 8), 32 * 4 * 7, #  TODO use 6x7 dimensions
                                  HIDDEN_CNN_LAYERS, momentum=0.9)
        game_root = Game()
        root = Node(game_root)
        mcts = MCTS(root, config={
            'ALPHA': 0.8,
            'CPUCT': 1,
            'EPSILON': 0.2
        }, model=model, state_encoder=StateEncoder(), action_encoder=ActionEncoder(DirectionResolver()))

        mcts.predict_state_value(game_root)
        mcts.evaluate_leaf(root)
