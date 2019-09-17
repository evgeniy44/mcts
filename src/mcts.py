from src.puct import PUCT


# import config
#
# from utils import setup_logger
# import loggers as lg


class MCTS:

    def __init__(self, root, config):
        self.root = root
        self.puct = PUCT(config['ALPHA'], config['EPSILON'], config['CPUCT'])

    def move_to_leaf(self):
        # lg.logger_mcts.info('------MOVING TO LEAF------')

        breadcrumbs = []
        current_node = self.root

        done = 0
        value = 0

        while not current_node.is_leaf():
            simulation_edge = self.puct.puct(current_node, is_root=current_node == self.root)
            # lg.logger_mcts.info('PLAYER TURN...%d', current_node.state.playerTurn)

            # lg.logger_mcts.info('action with highest Q + U...%d', simulation_edge.action)
            game = current_node.state.move(simulation_edge.action)
            value = 0
            done = game.is_over()
            if not done:
                value = 0
            else:
                value = game.get_winner()
            current_node = simulation_edge.out_node
            breadcrumbs.append(simulation_edge)

        # lg.logger_mcts.info('DONE...%d', done)

        return current_node, value, done, breadcrumbs


    def backFill(self, leaf, value, breadcrumbs):
        # lg.logger_mcts.info('------DOING BACKFILL------')

        current_player = leaf.state.playerTurn

        for edge in breadcrumbs:
            playerTurn = edge.player_turn
            if playerTurn == current_player:
                direction = 1
            else:
                direction = -1

            edge.stats['N'] = edge.stats['N'] + 1
            edge.stats['W'] = edge.stats['W'] + value * direction
            edge.stats['Q'] = edge.stats['W'] / edge.stats['N']
