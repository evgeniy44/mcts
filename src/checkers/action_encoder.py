import numpy as np
from sklearn.preprocessing import OneHotEncoder

from src.checkers.direction_resolver import NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST

DISTANCES_COUNT = 7

DIRECTIONS_COUNT = 4

POSITIONS_COUNT = 32


class ActionEncoder:

	def __init__(self, direction_resolver):
		self.direction_resolver = direction_resolver
		self.enc = OneHotEncoder()
		self.enc.fit(np.reshape(np.arange(0, POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT),
								newshape=(POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT, 1)))

	def convert_actions_to_values(self, actions):
		return list(map(self.convert_move_to_action_id, actions))

	def convert_move_to_action_id(self, action):
		direction, distance = self.direction_resolver.resolve_direction_and_distance(action)
		return action[0] + POSITIONS_COUNT * (direction - 1) + POSITIONS_COUNT * DIRECTIONS_COUNT * (distance - 1) - 1

	def convert_action_id_to_move_true_perspective(self, action_id, whose_turn):
		move = self.convert_action_id_to_move(action_id)
		if whose_turn == 1:
			return move
		move_true_perspective = []
		for position in move:
			move_true_perspective.append(33 - position)
		return move_true_perspective

	def convert_action_id_to_move(self, action_id):
		from_ = self.direction_resolver.get_coordinates(action_id % POSITIONS_COUNT + 1)
		direction = (action_id % (POSITIONS_COUNT * DIRECTIONS_COUNT)) // POSITIONS_COUNT + 1
		distance = action_id // (POSITIONS_COUNT * DIRECTIONS_COUNT) + 1

		if direction == NORTH_EAST:
			to = [from_[0] + distance, from_[1] + distance]
		elif direction == SOUTH_EAST:
			to = [from_[0] + distance, from_[1] - distance]
		elif direction == SOUTH_WEST:
			to = [from_[0] - distance, from_[1] - distance]
		elif direction == NORTH_WEST:
			to = [from_[0] - distance, from_[1] + distance]
		else:
			raise Exception("Unexpected direction: " + str(direction))

		return [self.__convert_coordinate_to_position_index(from_), self.__convert_coordinate_to_position_index(to)]

	def __convert_coordinate_to_position_index(self, coordinate):
		y = coordinate[1]
		x = 8 - coordinate[0]
		if y % 2 == 0:
			return y * 4 + x // 2
		else:
			return y * 4 + (x + 1) // 2

	def convert_action_to_one_hot(self, action):
		direction, distance = self.direction_resolver.resolve_direction_and_distance(action)
		value = action[0] + POSITIONS_COUNT * (direction - 1) + POSITIONS_COUNT * DIRECTIONS_COUNT * (distance - 1) - 1
		return self.enc.transform([[value]])

	def convert_one_hot_to_directed_action(self, encoded_action):
		value = np.asscalar(self.enc.inverse_transform(encoded_action))
		action = value % POSITIONS_COUNT + 1
		direction = (value % (POSITIONS_COUNT * DIRECTIONS_COUNT)) // POSITIONS_COUNT + 1
		distance = value // (POSITIONS_COUNT * DIRECTIONS_COUNT) + 1
		return action, direction, distance
