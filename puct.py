import numpy as np


class PUCT:

    def __init__(self, alpha, epsilon, cpuct):
        self.alpha = alpha  # noise weight
        self.epsilon = epsilon  # dirichlet distribution param, for root node only
        self.cpuct = cpuct  # exploration weight

    def puct(self, node, is_root=False):
        max_qu = -99999
        if is_root:
            noise = np.random.dirichlet([self.alpha] * len(node.edges))
            epsilon = self.epsilon
        else:
            epsilon = 0
            noise = [0] * len(node.edges)

        total_visits = 0
        for action, edge in enumerate(node.edges):
            total_visits = total_visits + edge.stats['N']

        for idx, (edge) in enumerate(node.edges):

            u = self.cpuct * \
                ((1 - epsilon) * edge.stats['P'] + epsilon * noise[idx]) * np.sqrt(total_visits) / (1 + edge.stats['N'])

            q = edge.stats['Q']
            #
            # lg.logger_mcts.info(
            #     'action: %d (%d)... N = %d, P = %f, noise = %f, adjP = %f, W = %f, Q = %f, U = %f, Q+U = %f'
            #     , action, action % 7, edge.stats['N'], np.round(edge.stats['P'], 6), np.round(noise[idx], 6),
            #     ((1 - epsilon) * edge.stats['P'] + epsilon * noise[idx])
            #     , np.round(edge.stats['W'], 6), np.round(Q, 6), np.round(U, 6), np.round(Q + U, 6))

            if q + u > max_qu:
                max_qu = q + u
                simulation_edge = edge

        return simulation_edge
