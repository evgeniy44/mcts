from unittest import TestCase

from src.checkers.direction_resolver import DirectionResolver

NORTH_EAST = 1
SOUTH_EAST = 2
SOUTH_WEST = 3
NORTH_WEST = 4


class TestDirectionResolver(TestCase):

	def test_resolve_direction_invalid_move(self):
		resolver = DirectionResolver()
		self.assertRaises(Exception, resolver.resolve_direction, [10, 11])

	def test_resolve_direction_north_east(self):
		resolver = DirectionResolver()
		self.assertEquals(resolver.resolve_direction([10, 17]), NORTH_EAST)
		self.assertEquals(resolver.resolve_direction([10, 14]), NORTH_EAST)

	def test_resolve_direction_south_east(self):
		resolver = DirectionResolver()
		self.assertEquals(resolver.resolve_direction([10, 6]), SOUTH_EAST)
		self.assertEquals(resolver.resolve_direction([10, 1]), SOUTH_EAST)

	def test_resolve_direction_south_west(self):
		resolver = DirectionResolver()
		self.assertEquals(resolver.resolve_direction([10, 7]), SOUTH_WEST)
		self.assertEquals(resolver.resolve_direction([10, 3]), SOUTH_WEST)

	def test_resolve_direction_north_west(self):
		resolver = DirectionResolver()
		self.assertEquals(resolver.resolve_direction([10, 15]), NORTH_WEST)
		self.assertEquals(resolver.resolve_direction([10, 19]), NORTH_WEST)

	def test_get_coordinates(self):
		resolver = DirectionResolver()
		x, y = resolver.get_coordinates(9)
		self.assertEqual(x, 6)
		self.assertEqual(y, 2)

	def test_get_coordinates_compressed(self):
		resolver = DirectionResolver()
		x, y = resolver.get_compressed_coordinates(9)
		self.assertEqual(x, 6)
		self.assertEqual(y, 1)

	def test_get_coordinates_compressed_2(self):
		resolver = DirectionResolver()
		x, y = resolver.get_compressed_coordinates(15)
		self.assertEqual(x, 3)
		self.assertEqual(y, 1)
