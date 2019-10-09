from src.agent import Agent
from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.state_encoder import StateEncoder
from src.match_conductor import MatchConductor
from src.model import Residual_CNN
import logging
from src.my_configs import config
import itertools
import operator


def run():
	logging.basicConfig(filename="tournament.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")
	result_table = {}
	versions_scores = {}
	round_n = 1

	for pair in itertools.combinations(config['TOURNAMENT_VERSIONS'], 2):
		logging.info("ROUND: " + str(round_n))

		player1 = read_agent(pair[0])
		player2 = read_agent(pair[1])

		logging.info("Starting match between versions: " + str(pair[0]) + " and " + str(pair[1]))

		match_conductor = MatchConductor()
		scores, _ = match_conductor.play_matches(player1, player2, config['EVAL_EPISODES'],
												 turns_until_tau0=0, memory=None)

		logging.info("Match result for pair: " + str(pair) + " is " + str(scores))
		result_table[pair] = scores

		if str(pair[0]) not in versions_scores:
			versions_scores[str(pair[0])] = scores['player' + str(pair[0])] + scores['drawn'] * 0.5
		else:
			versions_scores[str(pair[0])] = versions_scores[str(pair[0])] + scores['player' + str(pair[0])] + scores['drawn'] * 0.5

		if str(pair[1]) not in versions_scores:
			versions_scores[str(pair[1])] = scores['player' + str(pair[1])] + scores['drawn'] * 0.5
		else:
			versions_scores[str(pair[1])] = versions_scores[str(pair[1])] + scores['player' + str(pair[1])] + scores['drawn'] * 0.5

		round_n = round_n + 1

	sorted_result = sorted(versions_scores.items(), key=operator.itemgetter(1))
	logging.info("Tournament is over, results: " + str(sorted_result))


def read_agent(version):
	nn = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
							config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])
	logging.info('LOADING PLAYER MODEL VERSION ' + str(version) + '...')
	m_tmp = nn.read(version)
	nn.model.set_weights(m_tmp.get_weights())
	player = Agent(nn, ActionEncoder(DirectionResolver()), StateEncoder(), name='player' + str(version),
						   config=config)
	return player


if __name__ == "__main__":
	run()