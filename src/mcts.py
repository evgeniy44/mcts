import numpy as np

from src.edge import Edge
from src.node import Node
from src.puct import PUCT


class MCTS:

    def __init__(self, root, model, state_encoder, action_encoder, config):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.model = model
        self.root = root
        self.puct = PUCT(config['ALPHA'], config['EPSILON'], config['CPUCT'])
        self.tree = {}
        self.add_node(root)

    def move_to_leaf(self):
        breadcrumbs = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():
            simulation_edge = self.puct.puct(current_node, is_root=current_node == self.root)
            game = current_node.state.move_with_additional_jumps(
                self.action_encoder.convert_action_id_to_move_true_perspective(simulation_edge.action,
                                                                               simulation_edge.player_turn))
            done = game.is_over()
            if not done:
                value = 0
            else:
                value = game.get_winner_for_learning()
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)
        return current_node, value, done, breadcrumbs

    def backfill(self, leaf, value, breadcrumbs):
        current_player = leaf.state.whose_turn()

        for edge in breadcrumbs:
            player_turn = edge.player_turn
            if player_turn == current_player:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']

    def evaluate_leaf(self, leaf):
        value, probs, allowed_actions = self.predict_state_value(leaf.state)
        for action in allowed_actions:
            new_state = leaf.state.move_with_additional_jumps(
                self.action_encoder.convert_action_id_to_move_true_perspective(action, leaf.state.whose_turn()))
            if new_state.id not in self.tree:
                node = Node(new_state)
                self.add_node(node)
            else:
                node = self.tree[new_state.id]

            new_edge = Edge(leaf, node, probs[action], action)
            leaf.edges.append(new_edge)
        return value

    def predict_state_value(self, state):
        preds = self.model.predict(np.array([self.state_encoder.encode(state)]))
        value = preds[0][0][0]
        policies = preds[1][0]

        allowed_actions = self.action_encoder.convert_actions_to_values(
            state.get_possible_moves_from_current_player_perspective())

        mask = np.ones(policies.shape, dtype=bool)
        mask[np.array(allowed_actions)] = False
        policies[mask] = -100

        # SOFTMAX
        odds = np.exp(policies)
        probs = odds / np.sum(odds)

        return value, probs, allowed_actions

    def add_node(self, node):
        self.tree[node.id] = node
