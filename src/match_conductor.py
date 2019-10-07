import random

from src.checkers.game import Game
import logging

class MatchConductor:

    def __init__(self, initial_state=None):
        self.initial_state = initial_state

    def play_matches(self, player1, player2, episodes_count, turns_until_tau0, memory=None):

        scores = {player1.name: 0, "drawn": 0, player2.name: 0}

        for episode in range(episodes_count):
            logging.info("Running episode: " + str(episode))
            if self.initial_state is None:
                state = Game()
            else:
                state = self.initial_state

            done = 0
            turn = 0
            player1.mcts = None
            player2.mcts = None

            player1_starts = random.randint(0, 1) * 2 - 1

            if player1_starts == 1:
                players = {
                    1:
                        {
                            "agent": player1,
                            "name": player1.name
                        }
                    , 2:
                        {
                            "agent": player2,
                            "name": player2.name
                        }
                }
            else:
                players = {
                    1:
                        {
                            "agent": player2,
                            "name": player2.name
                        }
                    , 2:
                        {
                            "agent": player1,
                            "name": player1.name
                        }
                }

            while done == 0:
                turn = turn + 1
                if turn < turns_until_tau0:
                    action, pi, value = players[state.whose_turn()]['agent'].act(state, 1)
                else:
                    action, pi, value = players[state.whose_turn()]['agent'].act(state, 0)

                if memory is not None:
                    memory.commit_stmemory(state, pi)
                state = state.move(action)
                done = state.is_over()
                val = state.get_winner_for_learning()

                if state.is_over() == 1:
                    if memory is not None:
                        for move in memory.stmemory:
                            if move['state'].whose_turn() == state.whose_turn() and val != 0:
                                move['value'] = -1
                            elif val != 0:
                                move['value'] = 1
                            else:
                                move['value'] = 0
                        memory.commit_ltmemory()

                    if val != 0:
                        scores[players[state.opposite_turn()]['name']] = scores[players[state.opposite_turn()]['name']] + 1
                    else:
                        scores['drawn'] = scores['drawn'] + 1

        return scores, memory
