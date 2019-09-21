import numpy as np
from sklearn.preprocessing import OneHotEncoder

DISTANCES_COUNT = 7

DIRECTIONS_COUNT = 4

POSITIONS_COUNT = 32


class ActionEncoder:

	def __init__(self, direction_resolver):
		self.direction_resolver = direction_resolver
		self.enc = OneHotEncoder()
		self.enc.fit(np.reshape(np.arange(1, POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT + 1),
						   newshape=(POSITIONS_COUNT * DIRECTIONS_COUNT * DISTANCES_COUNT, 1)))

	def encode(self, action):
		direction, distance = self.direction_resolver.resolve_direction_and_distance(action)
		value = action[0] + POSITIONS_COUNT * (direction - 1) + POSITIONS_COUNT * DIRECTIONS_COUNT * (distance - 1) - 1
		return self.enc.transform([[value]])

	def decode(self, encoded_action):
		value = np.asscalar(self.enc.inverse_transform(encoded_action))
		action = value % POSITIONS_COUNT + 1
		direction = (value % (POSITIONS_COUNT * DIRECTIONS_COUNT)) // POSITIONS_COUNT + 1
		distance = value // (POSITIONS_COUNT * DIRECTIONS_COUNT) + 1
		return action, direction, distance

