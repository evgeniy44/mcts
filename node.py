class Node:

    def __init__(self, state):
        self.state = state
        self.edges = []

    def is_leaf(self):
        if len(self.edges) > 0:
            return False
        else:
            return True
