from src.agent import Agent
from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.state_encoder import StateEncoder
from src.match_conductor import MatchConductor
from src.memory import Memory
from src.model import Residual_CNN

config = {
	'ALPHA': 0.8,
	'CPUCT': 1,
	'EPSILON': 0.2,
	'ACTION_SIZE': 32 * 4 * 7,
	'MCTS_SIMULATIONS': 100,
	'REG_CONST': 0.0001,
	'LEARNING_RATE': 0.1,
	'EPISODES_COUNT': 20,
	'TURNS_UNTIL_TAU0': 10,
	'SCORING_THRESHOLD': 1.15,
	'EVAL_EPISODES': 20,
	'MEMORY_SIZE': 10000,
	'MOMENTUM': 0.9,
	'BATCH_SIZE': 256,
	'EPOCHS': 1,
	'TRAINING_LOOPS': 30,
	'HIDDEN_CNN_LAYERS': [
		{'filters': 75, 'kernel_size': (4, 4)}
		, {'filters': 75, 'kernel_size': (4, 4)}
		, {'filters': 75, 'kernel_size': (4, 4)}
		, {'filters': 75, 'kernel_size': (4, 4)}
		, {'filters': 75, 'kernel_size': (4, 4)}
		, {'filters': 75, 'kernel_size': (4, 4)}
	]
}

memory = Memory(config['MEMORY_SIZE'])

current_NN = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
						  config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])
best_NN = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
					   config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])


best_player_version = 0
best_NN.model.set_weights(current_NN.model.get_weights())

current_player = Agent(current_NN, ActionEncoder(DirectionResolver()), StateEncoder(), name='current_player', config=config)
best_player = Agent(best_NN, ActionEncoder(DirectionResolver()), StateEncoder(), name='best_player', config=config)

iteration = 0

while 1:

	iteration += 1
	print('ITERATION NUMBER ' + str(iteration))
	print('BEST PLAYER VERSION ' + str(best_player_version))

	match_conductor = MatchConductor()
	print('SELF PLAYING ' + str(config['EPISODES_COUNT']) + ' EPISODES...')
	_, memory = match_conductor.play_matches(best_player, best_player, config['EPISODES_COUNT'],
								  turns_until_tau0=config['TURNS_UNTIL_TAU0'], memory=memory)
	print('\n')

	memory.clear_stmemory()

	if len(memory.ltmemory) >= config['MEMORY_SIZE']:

		print('RETRAINING...')
		current_player.replay(memory.ltmemory)
		print('')

		print('TOURNAMENT...')
		scores, _ = match_conductor.play_matches(best_player, current_player, config['EVAL_EPISODES'],
												   turns_until_tau0=0, memory=None)
		print('\nSCORES')
		print(scores)
		# print('\nSTARTING PLAYER / NON-STARTING PLAYER SCORES')
		# print(sp_scores)
		# print(points)

		print('\n\n')

		if scores['current_player'] > scores['best_player'] * config['SCORING_THRESHOLD']:
			best_player_version = best_player_version + 1
			best_NN.model.set_weights(current_NN.model.get_weights())
			# best_NN.write(env.name, best_player_version)