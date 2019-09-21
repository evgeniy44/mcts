NORTH_EAST = 1
SOUTH_EAST = 2
SOUTH_WEST = 3
NORTH_WEST = 4


class DirectionResolver:

	def __init__(self):
		pass

	def resolve_direction_and_distance(self, move):
		x1, y1 = self.__get_coordinates(move[0])
		x2, y2 = self.__get_coordinates(move[1])

		if abs(x2 - x1) != abs(y2 - y1):
			raise Exception("looks like illegal move: " + str(move))
		if x2 > x1 and y2 > y1:
			return NORTH_EAST, abs(x2 - x1)
		if x2 > x1 and y2 < y1:
			return SOUTH_EAST, abs(x2 - x1)
		if x2 < x1 and y2 < y1:
			return SOUTH_WEST, abs(x2 - x1)
		if x2 < x1 and y2 > y1:
			return NORTH_WEST, abs(x2 - x1)

	def __get_coordinates(self, position):
		y = (position - 1) // 4
		x = (3 - ((position - 1) % 4)) * 2
		if y % 2 == 1:
			x = x + 1
		return x, y