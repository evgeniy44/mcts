import numpy as np

from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.state_encoder import StateEncoder
from src.mcts import MCTS
from src.node import Node
import random


class Agent:

	def __init__(self, model, action_encoder, state_encoder, name, config):
		self.state_encoder = state_encoder
		self.name = name
		self.action_encoder = action_encoder
		self.action_size = config['ACTION_SIZE']
		self.model = model
		self.mcts_simulations = config['MCTS_SIMULATIONS']
		self.mcts = None
		self.config = config
		self.train_overall_loss = []
		self.train_value_loss = []
		self.train_policy_loss = []

	def simulate(self):
		leaf, value, done, breadcrumbs = self.mcts.move_to_leaf()
		if not done:
			value = self.mcts.evaluate_leaf(leaf)
		self.mcts.backfill(leaf, value, breadcrumbs)

	def act(self, state, tau):
		self.prepare_mcts_for_next_action(state)
		for sim in range(self.mcts_simulations):
			self.simulate()
		pi, values = self.get_action_values()
		action, value = self.choose_action(pi, values, tau, state.whose_turn())
		return action, pi, value

	def prepare_mcts_for_next_action(self, state):
		if self.mcts is None or state.id not in self.mcts.tree:
			self.build_mcts(state)
		else:
			self.change_root_mcts(state)

	def build_mcts(self, state):
		self.root = Node(state)
		self.mcts = MCTS(self.root, self.model, self.state_encoder, self.action_encoder, config=self.config)

	def change_root_mcts(self, state):
		self.mcts.root = self.mcts.tree[state.id]

	def get_action_values(self):
		edges = self.mcts.root.edges
		pi = np.zeros(self.action_size, dtype=np.integer)
		values = np.zeros(self.action_size, dtype=np.float32)

		for edge in edges:
			pi[edge.action] = edge.stats['N']
			values[edge.action] = edge.stats['Q']

		pi = pi / (np.sum(pi) * 1.0)
		return pi, values

	def choose_action(self, pi, values, tau, whose_turn):
		if tau == 0:
			actions = np.argwhere(pi == max(pi))
			action = random.choice(actions)[0]
		else:
			action_idx = np.random.multinomial(1, pi)
			action = np.where(action_idx == 1)[0][0]

		value = values[action]

		return self.action_encoder.convert_action_id_to_move_true_perspective(action, whose_turn), value

	def replay(self, ltmemory):
		# lg.logger_mcts.info('******RETRAINING MODEL******')

		for i in range(self.config['TRAINING_LOOPS']):
			minibatch = random.sample(ltmemory, min(self.config['BATCH_SIZE'], len(ltmemory)))

			training_states = np.array([self.state_encoder.encode(row['state']) for row in minibatch])
			training_targets = {'value_head': np.array([row['value'] for row in minibatch])
				, 'policy_head': np.array([row['AV'] for row in minibatch])}

			fit = self.model.fit(training_states, training_targets, epochs=self.config['EPOCHS'], verbose=1, validation_split=0, batch_size=32)
			# lg.logger_mcts.info('NEW LOSS %s', fit.history)

			self.train_overall_loss.append(round(fit.history['loss'][self.config['EPOCHS'] - 1], 4))
			self.train_value_loss.append(round(fit.history['value_head_loss'][self.config['EPOCHS'] - 1], 4))
			self.train_policy_loss.append(round(fit.history['policy_head_loss'][self.config['EPOCHS'] - 1], 4))

		# plt.plot(self.train_overall_loss, 'k')
		# plt.plot(self.train_value_loss, 'k:')
		# plt.plot(self.train_policy_loss, 'k--')
		#
		# plt.legend(['train_overall_loss', 'train_value_loss', 'train_policy_loss'], loc='lower left')
		#
		# display.clear_output(wait=True)
		# display.display(pl.gcf())
		# pl.gcf().clear()
		# time.sleep(1.0)
		#
		# print('\n')
		# self.model.printWeightAverages()
