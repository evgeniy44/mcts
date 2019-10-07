import numpy as np
from sklearn.preprocessing import OneHotEncoder
import logging

from src.checkers.direction_resolver import NORTH_EAST, SOUTH_EAST, SOUTH_WEST, NORTH_WEST

DISTANCES_COUNT = 7

DIRECTIONS_COUNT = 4

POSITIONS_COUNT = 32


class ActionEncoder:

	def __init__(self, direction_resolver):
		self.action_ids_to_moves = {}
		self.moves_to_action_ids = {}
		self.direction_resolver = direction_resolver
		self.enc = OneHotEncoder(categories='auto')
		self.enc.fit(np.reshape(np.arange(0, POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT),
								newshape=(POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT, 1)))

	def convert_actions_to_values(self, actions): # TODO rename
		return list(map(self.convert_move_to_action_id, actions))

	def convert_move_to_action_id(self, move):
		if (move[0], move[1]) in self.moves_to_action_ids:
			logging.debug("Move: " + str(move) + " is already in cache, retrieving it")
			return self.moves_to_action_ids[(move[0], move[1])]
		logging.debug("Move: " + str(move) + " not in cache, calculating it")
		direction, distance = self.direction_resolver.resolve_direction_and_distance(move)
		action_id = move[0] + POSITIONS_COUNT * (direction - 1) + POSITIONS_COUNT * DIRECTIONS_COUNT * (
					distance - 1) - 1
		self.moves_to_action_ids[(move[0], move[1])] = action_id
		return action_id

	def convert_action_id_to_move_true_perspective(self, action_id, whose_turn):
		move = self.convert_action_id_to_move(action_id)
		if whose_turn == 1:
			return move
		move_true_perspective = []
		for position in move:
			move_true_perspective.append(33 - position)
		return move_true_perspective

	def convert_action_id_to_move(self, action_id):
		if action_id in self.action_ids_to_moves:
			logging.debug("action_id: " + str(action_id) + " is in cache, retrieving it from the cache")
			return self.action_ids_to_moves[action_id]
		logging.debug("action_id: " + str(action_id) + " not cached, calculating it")
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

		move = [self.__convert_coordinate_to_position_index(from_), self.__convert_coordinate_to_position_index(to)]
		self.action_ids_to_moves[action_id] = move
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
		direction, distance = self.direction_resolver.resolve_direction_and_distance(action)
		value = action[0] + POSITIONS_COUNT * (direction - 1) + POSITIONS_COUNT * DIRECTIONS_COUNT * (distance - 1) - 1
		return self.enc.transform([[value]])

	def convert_one_hot_to_directed_action(self, encoded_action):
		logging.warning("Shouldn't be used: convert_one_hot_to_directed_action")
		value = np.asscalar(self.enc.inverse_transform(encoded_action))
		action = value % POSITIONS_COUNT + 1
		direction = (value % (POSITIONS_COUNT * DIRECTIONS_COUNT)) // POSITIONS_COUNT + 1
		distance = value // (POSITIONS_COUNT * DIRECTIONS_COUNT) + 1
		return action, direction, distance
