config = {
    'ALPHA': 0.8,
    'CPUCT': 1,
    'EPSILON': 0.2,
    'ACTION_SIZE': 32 * 4,
    'MCTS_SIMULATIONS': 100,
    'REG_CONST': 0.0001,
    'LEARNING_RATE': 0.1,
    'EPISODES_COUNT': 30,
    'TURNS_UNTIL_TAU0': 10,
    'SCORING_THRESHOLD': 1.2,
    'EVAL_EPISODES': 10,
    'MEMORY_SIZE': 20000,
    'MOMENTUM': 0.9,
    'BATCH_SIZE': 256,
    'EPOCHS': 1,
    'TRAINING_LOOPS': 30,
    # 'INITIAL_MODEL_VERSION': 7,
    # 'INITIAL_MEMORY_VERSION': 18,
    'TOURNAMENT_VERSIONS': [1, 2, 3, 4, 5, 6, 7],
    'GUI_PLAYERS': [1, 7],
    'HIDDEN_CNN_LAYERS': [
        {'filters': 75, 'kernel_size': (4, 4)}
        , {'filters': 75, 'kernel_size': (4, 4)}
        , {'filters': 75, 'kernel_size': (4, 4)}
        , {'filters': 75, 'kernel_size': (4, 4)}
        , {'filters': 75, 'kernel_size': (4, 4)}
        , {'filters': 75, 'kernel_size': (4, 4)}
    ]
}