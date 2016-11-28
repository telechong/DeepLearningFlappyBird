ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
# OBSERVE = 100000.  # timesteps to observe before training
OBSERVE = 100.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
# REPLAY_MEMORY = 50000  # number of previous transitions to remember
REPLAY_MEMORY = 50  # number of previous transitions to remember
BATCH_SIZE = 32  # size of minibatch
FRAME_PER_ACTION = 1
UPDATE_TIME = 10000