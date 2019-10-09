from src.agent import Agent
from src.checkers.action_encoder import ActionEncoder
from src.checkers.direction_resolver import DirectionResolver
from src.checkers.state_encoder import StateEncoder
from src.match_conductor import MatchConductor
from src.memory import Memory
from src.model import Residual_CNN
import logging
import pickle
from src.my_configs import config

logging.basicConfig(filename="log.log", level=logging.INFO, format="%(asctime)s: %(levelname)s: %(message)s")

if "INITIAL_MEMORY_VERSION" not in config:
    memory = Memory(config['MEMORY_SIZE'])
else:
    logging.info("Loading memory version: " + str(config["INITIAL_MEMORY_VERSION"]))
    memory = pickle.load(open("memory/" + str(config["INITIAL_MEMORY_VERSION"]).zfill(4) + ".p", "rb"))

current_NN = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
                          config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])
best_NN = Residual_CNN(config['REG_CONST'], config['LEARNING_RATE'], (2, 4, 8), config['ACTION_SIZE'],
                       config['HIDDEN_CNN_LAYERS'], config['MOMENTUM'])

if "INITIAL_MODEL_VERSION" in config:
    best_player_version = config["INITIAL_MODEL_VERSION"]
    logging.info('LOADING MODEL VERSION ' + str(config["INITIAL_MODEL_VERSION"]) + '...')
    m_tmp = best_NN.read(best_player_version)
    current_NN.model.set_weights(m_tmp.get_weights())
    best_NN.model.set_weights(m_tmp.get_weights())
else:
    best_player_version = 0
    best_NN.model.set_weights(current_NN.model.get_weights())

current_player = Agent(current_NN, ActionEncoder(DirectionResolver()), StateEncoder(), name='current_player',
                       config=config)
best_player = Agent(best_NN, ActionEncoder(DirectionResolver()), StateEncoder(), name='best_player', config=config)

iteration = 0

while 1:

    iteration += 1
    logging.info('ITERATION NUMBER ' + str(iteration))
    logging.info('BEST PLAYER VERSION ' + str(best_player_version))

    match_conductor = MatchConductor()
    logging.info('SELF PLAYING ' + str(config['EPISODES_COUNT']) + ' EPISODES...')
    _, memory = match_conductor.play_matches(best_player, best_player, config['EPISODES_COUNT'],
                                             turns_until_tau0=config['TURNS_UNTIL_TAU0'], memory=memory)
    memory.clear_stmemory()

    logging.info("Current memory size: " + str(len(memory.ltmemory)))
    if len(memory.ltmemory) >= config['MEMORY_SIZE']:

        if iteration % 3 == 0:
            pickle.dump(memory, open("memory/" + str(iteration).zfill(4) + ".p", "wb"))

        logging.info('RETRAINING...')
        current_player.replay(memory.ltmemory)

        logging.info('TOURNAMENT...')
        scores, _ = match_conductor.play_matches(best_player, current_player, config['EVAL_EPISODES'],
                                                 turns_until_tau0=0, memory=None)
        logging.info('SCORES')
        logging.info(scores)

        if scores['current_player'] > scores['best_player'] * config['SCORING_THRESHOLD']:
            best_player_version = best_player_version + 1
            logging.info("Updating best player to the new version: " + str(best_player_version))
            best_NN.model.set_weights(current_NN.model.get_weights())
            best_NN.write(best_player_version)
