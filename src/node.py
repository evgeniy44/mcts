class Node:

    def __init__(self, state):
        self.state = state
        self.edges = []
        self.id = state.id

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True
