import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging

from src.checkers.direction_resolver import NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST

DIRECTIONS_COUNT = 4

POSITIONS_COUNT = 32


class ActionEncoder:

	def __init__(self, direction_resolver):
		self.moves_to_action_ids = {}
		self.direction_resolver = direction_resolver
		self.enc = OneHotEncoder(categories='auto')
		self.enc.fit(np.reshape(np.arange(0, POSITIONS_COUNT * DIRECTIONS_COUNT),
								newshape=(POSITIONS_COUNT * DIRECTIONS_COUNT, 1)))

	def convert_moves_to_action_ids(self, actions):
		return list(map(self.convert_move_to_action_id, actions))

	def convert_move_to_action_id(self, move):
		if (move[0], move[1]) in self.moves_to_action_ids:
			logging.debug("Move: " + str(move) + " is already in cache, retrieving it")
			return self.moves_to_action_ids[(move[0], move[1])]
		logging.debug("Move: " + str(move) + " not in cache, calculating it")
		direction = self.direction_resolver.resolve_direction(move)
		action_id = move[0] + POSITIONS_COUNT * (direction - 1) - 1
		self.moves_to_action_ids[(move[0], move[1])] = action_id
		return action_id

	def convert_action_id_to_position_and_direction(self, action_id):
		logging.debug("action_id: " + str(action_id) + " not cached, calculating it")
		position = action_id % POSITIONS_COUNT + 1
		direction = (action_id % (POSITIONS_COUNT * DIRECTIONS_COUNT)) // POSITIONS_COUNT + 1
		return position, direction

	def convert_action_id_to_true_position_and_direction(self, action_id, whose_turn):
		position, direction = self.convert_action_id_to_position_and_direction(action_id)
		if whose_turn == 1:
			return position, direction
		direction = (direction + 2) % 4
		if direction == 0:
			direction = 4
		position = 33 - position
		return position, direction

	def convert_direction_and_distance_to_move(self, position, direction, distance):
		from_ = self.direction_resolver.get_coordinates(position)
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
		move = [self.__convert_coordinate_to_position_index(from_), self.__convert_coordinate_to_position_index(to)]
		return move

	def __convert_coordinate_to_position_index(self, coordinate):
		y = coordinate[1]
		x = 8 - coordinate[0]
		if y % 2 == 0:
			return y * 4 + x // 2
		else:
			return y * 4 + (x + 1) // 2

	def convert_action_to_one_hot(self, action):
		logging.warning("Shouldn't be used: convert_action_to_one_hot")
		direction = self.direction_resolver.resolve_direction(action)
		value = action[0] + POSITIONS_COUNT * (direction - 1) - 1
		return self.enc.transform([[value]])
