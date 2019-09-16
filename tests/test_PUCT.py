from unittest import TestCase

from edge import Edge
from node import Node
from puct import PUCT

import numpy as np


class TestPUCT(TestCase):

    def test_puct_root_node(self):
        np.random.seed(1)
        puct = PUCT(0.8, 0.2, 1)
        parent_node = Node({'id': 0})
        parent_node.edges.append(Edge(parent_node, Node({'id': 1}), 0.14285715, 35))
        parent_node.edges.append(Edge(parent_node, Node({'id': 2}), 0.14285715, 36))
        parent_node.edges.append(Edge(parent_node, Node({'id': 3}), 0.14285715, 37))
        parent_node.edges.append(Edge(parent_node, Node({'id': 4}), 0.14285715, 38))
        parent_node.edges.append(Edge(parent_node, Node({'id': 5}), 0.14285715, 39))
        parent_node.edges.append(Edge(parent_node, Node({'id': 6}), 0.14285715, 40))
        parent_node.edges.append(Edge(parent_node, Node({'id': 7}), 0.14285715, 41))
        simulation_edge = puct.puct(parent_node, is_root=True)

        self.assertEquals(simulation_edge.action, 35)
        self.assertEquals(simulation_edge.inNode.state['id'], 0)
        self.assertEquals(simulation_edge.outNode.state['id'], 1)

    def test_puct_non_root_node(self):
        np.random.seed(1)
        puct = PUCT(0.8, 0.2, 1)
        parent_node = Node({'id': 0})
        parent_node.edges.append(Edge(parent_node, Node({'id': 1}), 0.14805108, 29))
        parent_node.edges.append(Edge(parent_node, Node({'id': 2}), 0.14307857, 35))
        parent_node.edges.append(Edge(parent_node, Node({'id': 3}), 0.14475949, 37))
        parent_node.edges.append(Edge(parent_node, Node({'id': 4}), 0.1387326, 38))
        parent_node.edges.append(Edge(parent_node, Node({'id': 5}), 0.14208362, 39))
        parent_node.edges.append(Edge(parent_node, Node({'id': 6}), 0.14188258, 40))
        parent_node.edges.append(Edge(parent_node, Node({'id': 7}), 0.14141211, 41))
        simulation_edge = puct.puct(parent_node, is_root=False)

        self.assertEquals(simulation_edge.action, 29)
        self.assertEquals(simulation_edge.inNode.state['id'], 0)
        self.assertEquals(simulation_edge.outNode.state['id'], 1)

    def test_puct_non_root_node_exploration(self):
        np.random.seed(1)
        puct = PUCT(0.8, 0.2, 1)
        parent_node = Node({'id': 0})
        edge1 = Edge(parent_node, Node({'id': 1}), 0.14805108, 29)
        edge1.stats['N'] = 100
        parent_node.edges.append(edge1)

        edge2 = Edge(parent_node, Node({'id': 2}), 0.14307857, 35)
        edge2.stats['N'] = 100
        parent_node.edges.append(edge2)

        edge3 = Edge(parent_node, Node({'id': 3}), 0.14475949, 37)
        edge3.stats['N'] = 100
        parent_node.edges.append(edge3)

        edge4 = Edge(parent_node, Node({'id': 4}), 0.1387326, 38)
        edge4.stats['N'] = 10
        parent_node.edges.append(edge4)

        edge5 = Edge(parent_node, Node({'id': 5}), 0.14208362, 39)
        edge5.stats['N'] = 100
        parent_node.edges.append(edge5)

        edge6 = Edge(parent_node, Node({'id': 6}), 0.14188258, 40)
        edge6.stats['N'] = 100
        parent_node.edges.append(edge6)

        edge7 = Edge(parent_node, Node({'id': 7}), 0.14141211, 41)
        edge7.stats['N'] = 100
        parent_node.edges.append(edge7)

        simulation_edge = puct.puct(parent_node, is_root=False)

        self.assertEquals(simulation_edge.action, 38)
        self.assertEquals(simulation_edge.inNode.state['id'], 0)
        self.assertEquals(simulation_edge.outNode.state['id'], 4)
