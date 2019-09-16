class Edge:

    def __init__(self, in_node, out_node, probability, action):
        # self.id = in_node.state.id + '|' + out_node.state.id
        self.in_node = in_node
        self.out_node = out_node
        # self.playerTurn = in_node.state.playerTurn
        self.action = action

        self.stats = {
            'N': 0,
            'W': 0,
            'Q': 0,
            'P': probability,
        }
