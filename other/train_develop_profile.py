"""
End to end training of my neural network model.

The training routine has three key phases

- Evaluation through MCTS
- Data generation through MCTS
- Neural network training
"""
import numpy as np
from collections import defaultdict, deque, Counter, namedtuple
import itertools
import warnings
import os, psutil # useful for memory management
from datetime import datetime

from mcts_nn_cube import State, MCTSAgent

def init_yappi():
    OUT_FILE = '/tmp/pants'

    import atexit
    import yappi

    print('[YAPPI START]')
    yappi.set_clock_type('wall')
    yappi.start()

    @atexit.register
    def finish_yappi():
        print('[YAPPI STOP]')

        yappi.stop()

        print('[YAPPI WRITE]')

        stats = yappi.get_func_stats()

        for stat_type in ['pstat', 'callgrind', 'ystat']:
            print('writing {}.{}'.format(OUT_FILE, stat_type))
            #stats.save('{}.{}'.format(OUT_FILE, stat_type), type=stat_type)

        print('\n[YAPPI FUNC_STATS]')

        print('writing {}.func_stats'.format(OUT_FILE))
        stats.print_all()
        #with open('{}.func_stats'.format(OUT_FILE), 'wb') as fh:
        #    stats.print_all(out=fh)

        print('\n[YAPPI THREAD_STATS]')

        print('writing {}.thread_stats'.format(OUT_FILE))
        tstats = yappi.get_thread_stats()
        tstats.print_all()
        #with open('{}.thread_stats'.format(OUT_FILE), 'wb') as fh:
        #    tstats.print_all(out=fh)

        print('[YAPPI OUT]')

# this keeps track of the training runs, including the older versions that we are extending
VERSIONS = ["v0.9.2.1", "v0.9.2"]

# memory management
MY_PROCESS = psutil.Process(os.getpid())
def memory_used():
    return MY_PROCESS.memory_info().rss

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

"""
# for putting the cube in 3D
x3d = \
np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2,
          3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 3,
          2, 1, 3, 2, 1, 3, 2, 1])

y3d = \
np.array([1, 1, 1, 2, 2, 2, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 2, 2,
          2, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1,
          1, 1, 2, 2, 2, 3, 3, 3])

z3d = \
np.array([3, 2, 1, 3, 2, 1, 3, 2, 1, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 0,
          0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 4,
          4, 4, 4, 4, 4, 4, 4, 4])

neighbors = \
np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  0, -1,  4,  3,
           -1, 12,  9, -1, -1, -1, 47, -1, -1, 50],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  1,  0,  5,  4,
            3, 15, 12,  9, -1, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  1, -1,  5,
            4, -1, 15, 12, 18, -1, -1, 21, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1,  1,  0, -1,  4,  3, -1,  7,  6,
           -1, -1, -1, 47, -1, -1, 50, -1, -1, 53],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  1,  0,  5,  4,  3,  8,  7,
            6, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  2,  1, -1,  5,  4, -1,  8,
            7, 18, -1, -1, 21, -1, -1, 24, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1,  4,  3, -1,  7,  6, -1, -1, -1,
           -1, -1, -1, 50, -1, -1, 53, 30, 33, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1,  5,  4,  3,  8,  7,  6, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, 27, 30, 33],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5,  4, -1,  8,  7, -1, -1,
           -1, 21, -1, -1, 24, -1, -1, -1, 27, 30],
          [-1, -1, -1, -1, -1, -1,  1,  0, -1, -1, -1, -1, 12,  9, -1, -1, -1,
           47, -1, -1, -1, 13, 10, -1, -1, -1, 46],
          [-1, -1, -1, 12,  9, -1, -1, -1, 47, -1, -1, -1, 13, 10, -1, -1, -1,
           46, -1, -1, -1, 14, 11, -1, -1, -1, 45],
          [-1, -1, -1, 13, 10, -1, -1, -1, 46, -1, -1, -1, 14, 11, -1, -1, -1,
           45, -1, -1, -1, -1, -1, -1, 37, 38, -1],
          [-1, -1, -1, -1, -1, -1,  2,  1,  0, -1, -1, -1, 15, 12,  9, -1, -1,
           -1, -1, -1, -1, 16, 13, 10, -1, -1, -1],
          [-1, -1, -1, 15, 12,  9, -1, -1, -1, -1, -1, -1, 16, 13, 10, -1, -1,
           -1, -1, -1, -1, 17, 14, 11, -1, -1, -1],
          [-1, -1, -1, 16, 13, 10, -1, -1, -1, -1, -1, -1, 17, 14, 11, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, 36, 37, 38],
          [-1, -1, -1, -1, -1, -1, -1,  2,  1, -1, -1, -1, -1, 15, 12, 18, -1,
           -1, -1, -1, -1, -1, 16, 13, 19, -1, -1],
          [-1, -1, -1, -1, 15, 12, 18, -1, -1, -1, -1, -1, -1, 16, 13, 19, -1,
           -1, -1, -1, -1, -1, 17, 14, 20, -1, -1],
          [-1, -1, -1, -1, 16, 13, 19, -1, -1, -1, -1, -1, -1, 17, 14, 20, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, 36, 37],
          [-1, -1, -1, -1, -1,  2, -1, -1,  5, -1, -1, 15, -1, 18, -1, -1, 21,
           -1, -1, -1, 16, -1, 19, -1, -1, 22, -1],
          [-1, -1, 15, -1, 18, -1, -1, 21, -1, -1, -1, 16, -1, 19, -1, -1, 22,
           -1, -1, -1, 17, -1, 20, -1, -1, 23, -1],
          [-1, -1, 16, -1, 19, -1, -1, 22, -1, -1, -1, 17, -1, 20, -1, -1, 23,
           -1, -1, -1, -1, -1, -1, 36, -1, -1, 39],
          [-1, -1,  2, -1, -1,  5, -1, -1,  8, -1, 18, -1, -1, 21, -1, -1, 24,
           -1, -1, 19, -1, -1, 22, -1, -1, 25, -1],
          [-1, 18, -1, -1, 21, -1, -1, 24, -1, -1, 19, -1, -1, 22, -1, -1, 25,
           -1, -1, 20, -1, -1, 23, -1, -1, 26, -1],
          [-1, 19, -1, -1, 22, -1, -1, 25, -1, -1, 20, -1, -1, 23, -1, -1, 26,
           -1, -1, -1, 36, -1, -1, 39, -1, -1, 42],
          [-1, -1,  5, -1, -1,  8, -1, -1, -1, -1, 21, -1, -1, 24, -1, -1, -1,
           27, -1, 22, -1, -1, 25, -1, -1, -1, 28],
          [-1, 21, -1, -1, 24, -1, -1, -1, 27, -1, 22, -1, -1, 25, -1, -1, -1,
           28, -1, 23, -1, -1, 26, -1, -1, -1, 29],
          [-1, 22, -1, -1, 25, -1, -1, -1, 28, -1, 23, -1, -1, 26, -1, -1, -1,
           29, -1, -1, 39, -1, -1, 42, -1, -1, -1],
          [-1,  8,  7, -1, -1, -1, -1, -1, -1, 24, -1, -1, -1, 27, 30, -1, -1,
           -1, 25, -1, -1, -1, 28, 31, -1, -1, -1],
          [24, -1, -1, -1, 27, 30, -1, -1, -1, 25, -1, -1, -1, 28, 31, -1, -1,
           -1, 26, -1, -1, -1, 29, 32, -1, -1, -1],
          [25, -1, -1, -1, 28, 31, -1, -1, -1, 26, -1, -1, -1, 29, 32, -1, -1,
           -1, -1, 42, 43, -1, -1, -1, -1, -1, -1],
          [ 8,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, 27, 30, 33, -1, -1,
           -1, -1, -1, -1, 28, 31, 34, -1, -1, -1],
          [-1, -1, -1, 27, 30, 33, -1, -1, -1, -1, -1, -1, 28, 31, 34, -1, -1,
           -1, -1, -1, -1, 29, 32, 35, -1, -1, -1],
          [-1, -1, -1, 28, 31, 34, -1, -1, -1, -1, -1, -1, 29, 32, 35, -1, -1,
           -1, 42, 43, 44, -1, -1, -1, -1, -1, -1],
          [ 7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, 53, 30, 33, -1, -1, -1,
           -1, -1, -1, 52, 31, 34, -1, -1, -1, -1],
          [-1, -1, 53, 30, 33, -1, -1, -1, -1, -1, -1, 52, 31, 34, -1, -1, -1,
           -1, -1, -1, 51, 32, 35, -1, -1, -1, -1],
          [-1, -1, 52, 31, 34, -1, -1, -1, -1, -1, -1, 51, 32, 35, -1, -1, -1,
           -1, 43, 44, -1, -1, -1, -1, -1, -1, -1],
          [-1, 17, 14, 20, -1, -1, 23, -1, -1, -1, -1, -1, -1, 36, 37, -1, 39,
           40, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [17, 14, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, 36, 37, 38, 39, 40,
           41, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [14, 11, -1, -1, -1, 45, -1, -1, 48, -1, -1, -1, 37, 38, -1, 40, 41,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [20, -1, -1, 23, -1, -1, 26, -1, -1, -1, 36, 37, -1, 39, 40, -1, 42,
           43, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, 36, 37, 38, 39, 40, 41, 42, 43,
           44, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, 45, -1, -1, 48, -1, -1, 51, 37, 38, -1, 40, 41, -1, 43, 44,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [23, -1, -1, 26, -1, -1, -1, 29, 32, -1, 39, 40, -1, 42, 43, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, 29, 32, 35, 39, 40, 41, 42, 43, 44, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, 48, -1, -1, 51, 32, 35, -1, 40, 41, -1, 43, 44, -1, -1, -1,
           -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [10, -1, -1, -1, 46, -1, -1, 49, -1, 11, -1, -1, -1, 45, -1, -1, 48,
           -1, -1, -1, -1, 38, -1, -1, 41, -1, -1],
          [ 9, -1, -1, -1, 47, -1, -1, 50, -1, 10, -1, -1, -1, 46, -1, -1, 49,
           -1, 11, -1, -1, -1, 45, -1, -1, 48, -1],
          [-1, -1, -1,  0, -1, -1,  3, -1, -1,  9, -1, -1, -1, 47, -1, -1, 50,
           -1, 10, -1, -1, -1, 46, -1, -1, 49, -1],
          [-1, 46, -1, -1, 49, -1, -1, 52, -1, -1, 45, -1, -1, 48, -1, -1, 51,
           -1, 38, -1, -1, 41, -1, -1, 44, -1, -1],
          [-1, 47, -1, -1, 50, -1, -1, 53, -1, -1, 46, -1, -1, 49, -1, -1, 52,
           -1, -1, 45, -1, -1, 48, -1, -1, 51, -1],
          [ 0, -1, -1,  3, -1, -1,  6, -1, -1, -1, 47, -1, -1, 50, -1, -1, 53,
           -1, -1, 46, -1, -1, 49, -1, -1, 52, -1],
          [-1, 49, -1, -1, 52, -1, 34, -1, -1, -1, 48, -1, -1, 51, -1, 35, -1,
           -1, 41, -1, -1, 44, -1, -1, -1, -1, -1],
          [-1, 50, -1, -1, 53, -1, 33, -1, -1, -1, 49, -1, -1, 52, -1, 34, -1,
           -1, -1, 48, -1, -1, 51, -1, 35, -1, -1],
          [ 3, -1, -1,  6, -1, -1, -1, -1, -1, -1, 50, -1, -1, 53, -1, 33, -1,
           -1, -1, 49, -1, -1, 52, -1, 34, -1, -1]])
"""

class GameAgent():
    def __init__(self, game_id):
        self.game_id = game_id
        self.self_play_stats=defaultdict(list)
        self.game_stats=defaultdict(list)
        self.data_states = []
        self.data_policies = []
        self.data_values = []
        self.counter=0
        self.done=False
        self.win=False
        # can attach other attributes as needed

class BatchGameAgent():
    """
    Handles the steps of the games, including batch games.
    """
    def __init__(self, model, max_steps, max_depth, max_game_length, transposition_table, decay, exploration):
        self.game_agents = deque()
        self.model = model
        self.max_depth = max_depth
        self.max_steps = max_steps
        self.max_game_length = max_game_length
        self.transposition_table = transposition_table
        self.exploration = exploration
        self.decay = decay

    def is_empty(self):
        return not bool(self.game_agents)

    def append_states(self, state_info_iter):
        for game_id, state, distance, distance_level in state_info_iter:
            mcts = MCTSAgent(self.model.function, 
                             state.copy(), 
                             max_depth = self.max_depth, 
                             transposition_table = self.transposition_table.copy(),
                             c_puct = self.exploration,
                             gamma = self.decay)
            
            game_agent = GameAgent(game_id)
            game_agent.mcts = mcts
            game_agent.distance = distance
            game_agent.distance_level = distance_level

            self.game_agents.append(game_agent)

    def run_game_agent_one_step(self, game_agent):
        mcts = game_agent.mcts
        mcts.search(steps=self.max_steps)

        # reduce the max batch size to prevent the worker from blocking
        self.model.set_max_batch_size(self.model.get_max_batch_size() - 1)

    def process_completed_step(self, game_agent):
        mcts = game_agent.mcts
            
        # find next state
        probs = mcts.action_probabilities(inv_temp = 10)
        action = np.argmax(probs)
        #action = np.random.choice(12, p=probs)

        shortest_path = game_agent.mcts.stats('shortest_path')

        # record stats
        game_agent.self_play_stats['_game_id'].append(game_agent.game_id)
        game_agent.self_play_stats['_step_id'].append(game_agent.counter)
        game_agent.self_play_stats['shortest_path'].append(shortest_path)
        game_agent.self_play_stats['action'].append(action)
        game_agent.self_play_stats['value'].append(mcts.stats('value'))

        game_agent.self_play_stats['prior'].append(mcts.stats('prior'))
        game_agent.self_play_stats['prior_dirichlet'].append(mcts.stats('prior_dirichlet'))
        game_agent.self_play_stats['visit_counts'].append(mcts.stats('visit_counts'))
        game_agent.self_play_stats['total_action_values'].append(mcts.stats('total_action_values'))

        # training data (also recorded in stats)
        game_agent.data_states.append(mcts.initial_node.state.input_array())
        
        policy = mcts.action_probabilities(inv_temp = 10)
        game_agent.data_policies.append(policy)
        game_agent.self_play_stats['updated_policy'].append(policy)
        
        game_agent.data_values.append(0) # updated if game is success
        game_agent.self_play_stats['updated_value'].append(0)

        # prepare for next state
        game_agent.counter += 1 
        #if shortest_path < 0:
        #    print("(DB) no path")
        if (game_agent.counter > 1 and shortest_path < 0) or game_agent.counter >= self.max_game_length:
            game_agent.win = False
            game_agent.done = True
        else:
            mcts.advance_to_action(action)
            if mcts.is_terminal():
                game_agent.win = True
                game_agent.done = True

    def run_one_step_with_threading(self):
        import threading
        # start threads
        self.model.set_max_batch_size(len(self.game_agents))

        threads = []
        for game_agent in self.game_agents:
            t = threading.Thread(target=self.run_game_agent_one_step, args=(game_agent, ))
            t.start()
            threads.append(t)

        # wait for threads to finish
        for t in threads:
            t.join()

        for game_agent in self.game_agents:
            self.process_completed_step(game_agent)

    def run_one_step(self):
        for game_agent in self.game_agents:

            mcts = game_agent.mcts
            mcts.search(steps=self.max_steps)
            
            self.process_completed_step(game_agent)

    def finished_game_results(self):
        for _ in range(len(self.game_agents)):
            game_agent = self.game_agents.popleft()

            if not game_agent.done:
                self.game_agents.append(game_agent)
            else:
                if game_agent.win:
                    value = 1
                    for i in range(game_agent.counter):
                        value *= self.decay
                        game_agent.data_values[-(i+1)] = value
                        game_agent.self_play_stats['updated_value'][-(i+1)] = value
          
                # record game stats
                game_agent.game_stats['_game_id'].append(game_agent.game_id)
                game_agent.game_stats['distance_level'].append(game_agent.distance_level)
                game_agent.game_stats['training_distance'].append(game_agent.distance)
                game_agent.game_stats['max_game_length'].append(self.max_game_length)
                game_agent.game_stats['win'].append(game_agent.win)
                game_agent.game_stats['total_steps'].append(game_agent.counter if game_agent.win else -1)

                yield game_agent

class TrainingAgent():
    """
    This agent handles all the details of the training.
    """
    def __init__(self):
        import models 

        # Threading
        self.multithreaded = True

        # Model (NN) parameters (fixed)
        self.checkpoint_model = models.ConvModel2D3D() # this doesn't build and/or load the model yet
        self.best_model = models.ConvModel2D3D()       # this doesn't build and/or load the model yet
        if self.multithreaded:
            self.checkpoint_model.multithreaded = True
            self.best_model.multithreaded = True
        self.learning_rate = .001

        # MCTS parameters (fixed)
        self.max_depth = 900
        self.max_steps = 1600
        self.use_prebuilt_transposition_table = False
        self.decay = 0.95 # gamma
        self.exploration = 1. #c_puct
        self.prebuilt_transposition_table = None # built later

        # Validation flags
        self.validate_training_data = True

        # Training parameters (fixed)
        self.batch_size = 32
        self.games_per_generation = 100
        self.starting_distance = 1
        self.min_distance = 1
        self.win_rate_target = .5
        self.max_game_length = 100
        self.prev_generations_used_for_training = 8
        self.training_sample_ratio = 1/self.prev_generations_used_for_training
        self.games_per_evaluation = 128

        # Training parameters preserved between generations
        self.training_distance_level = float(self.starting_distance)
        self.recent_wins = Counter()
        self.recent_games = Counter()
        self.checkpoint_training_distance_level = float(self.starting_distance)
        self.checkpoint_recent_wins = Counter()
        self.checkpoint_recent_games = Counter()

        # Training parameters (dynamic)
        self.game_number = 0
        self.self_play_start = None # date and time (utc)
        self.self_play_end = None
        self.training_start = None
        self.training_end = None

        # Evaluation parameters (dynamic)
        self.generation = 0
        self.best_generation = 0

        # Self play stats
        # These are functionally data tables implemented as a dictionary of lists
        # The keys are the column names.  This makes it easy to change the stats I am recording.
        self.self_play_stats = defaultdict(list)
        self.game_stats = defaultdict(list)
        self.training_stats = defaultdict(list)
        self.generation_stats = defaultdict(list)

        # Training data
        self.training_data_states = []
        self.training_data_policies = []
        self.training_data_values = []

    def __remove_me__starting_model(self):
        """
        Build and return a new neural network using the current model architecture
        """
        import numpy as np
        from keras.models import Model
        from keras.layers import Conv2D, Input, BatchNormalization, Dense, Flatten, Activation, add, Lambda, Reshape
        from keras.optimizers import Adam
        from keras.losses import categorical_crossentropy
        from keras.regularizers import l2
        import keras.backend as K

        import tensorflow as tf

        neighbors[neighbors == -1] = 54
        
        def special_cube_conv(in_tensor, filter_size):
            """
            Takes in a None (samples) x 54 x ? (filters) tensor.

            It embedds it into 5 x 5 grid, and does a 3D convolution
            using only the nodes in the orginal embedding.

            To speed things up, it actually does the folowing:
            - pads the end with a zero (in the last dimension):
                None (samples) x 55 x ? (filters) (neighbors)
            - align neighbors to get an output of dim:
                None (samples) x 54 x 27 x ? (filters) (neighbors)
            - 2d convolution with filter (1, 27) and no padding to get an output of dim:
                None (samples) x 54 x filter_size
            - reshape to remove last dimension:
                None (samples) x filter_size x 54
            """ 
            print("in    ", in_tensor.shape)
            # pad (output dim: None x 55 x ?)
            padded = Lambda(lambda x: K.temporal_padding(x, (0, 1)))(in_tensor) # just pad end
            print("padded", padded.shape)
            # align neighbors (output dim: None x 54 x 27 x ?)
            #aligned = K.gather(padded, neighbors)
            #aligned = padded[ neighbors[np.newaxis].astype(np.int32), :]
            aligned = Lambda(lambda x: tf.gather(x, neighbors, axis=1))(padded)
            print("align ", aligned.shape)
            # 2D convolution in one axis (output dim: None x 54 x 1 x filter_size)
            conv = Conv2D(filter_size, kernel_size=(1, 27), 
                          strides=(1, 1), 
                          padding='valid', 
                          data_format="channels_last",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(aligned)

            print("conv  ", conv.shape)
            # reshape (output dim: None x 54 x filter_size)
            out_tensor = Lambda(lambda x: K.squeeze(x, axis=2))(conv)

            return out_tensor

        def conv_block(in_tensor, filter_size):
            conv = special_cube_conv(in_tensor, filter_size)
            batch = BatchNormalization(axis=1)(conv)
            relu = Activation('relu')(batch)

            return relu

        def residual_block(in_tensor, filter_size):
            conv1 = special_cube_conv(in_tensor, filter_size)
            batch1 = BatchNormalization(axis=1)(conv1)
            relu1 = Activation('relu')(batch1)

            conv2 = special_cube_conv(relu1, filter_size)
            batch2 = BatchNormalization(axis=1)(conv2)

            combine = add([batch2, in_tensor])
            relu = Activation('relu')(combine)

            return relu

        def policy_block(in_tensor, filter_size, hidden_size):
            conv = conv_block(in_tensor, filter_size=filter_size)
            flat = Flatten()(conv)
            hidden = Dense(hidden_size, activation='relu',
                           kernel_regularizer=l2(0.001), 
                           bias_regularizer=l2(0.001))(flat)
            output = Dense(12, activation='softmax',
                           kernel_regularizer=l2(0.001), 
                           bias_regularizer=l2(0.001),
                           name='policy_output')(hidden)
            return output

        def value_block(in_tensor, filter_size, hidden_size):
            conv = conv_block(in_tensor, filter_size=filter_size)
            flat = Flatten()(conv)
            hidden = Dense(hidden_size, activation='relu',
                           kernel_regularizer=l2(0.001), 
                           bias_regularizer=l2(0.001))(flat)
            output = Dense(1, activation='sigmoid',
                           kernel_regularizer=l2(0.001), 
                           bias_regularizer=l2(0.001),
                           name='value_output')(hidden)
            return output

        # the network
        state_input = Input(shape=(54, 6), name='state_input')
        
        # convolutional
        block = conv_block(state_input, filter_size=64)

        # multiple residuals
        block = residual_block(block, filter_size=64)
        block = residual_block(block, filter_size=64)
        block = residual_block(block, filter_size=64)
        block = residual_block(block, filter_size=64)

        # policy head
        policy_output = policy_block(block, filter_size=64, hidden_size=64)

        # value head
        value_output = value_block(block, filter_size=64, hidden_size=64)

        # combine
        model = Model(inputs=state_input, outputs=[policy_output, value_output])
        model.compile(loss={'policy_output': categorical_crossentropy, 
                            'value_output': 'mse'},
                      loss_weights={'policy_output': 1., 'value_output': 1.},
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def __remove_me__build_model_policy_value(self, model, max_cache_size=100000):
        from collections import OrderedDict
        cache = OrderedDict()
        #from tensorflow.python.keras import backend as K
        #get_output = K.function([model.input, K.learning_phase()], [model.output[0], model.output[1]])
        def model_policy_value(input_array):
            key = input_array.tobytes()
            if key in cache:
                cache.move_to_end(key, last=True)
                return cache[key]
            
            input_array = input_array.reshape((-1, 54, 6))
            #input_array = np.rollaxis(input_array, 2, 1)
            
            policy, value = model.predict(input_array)
            #policy, value = get_output([input_array, 0])
            policy = policy.reshape((12,))
            value = value[0, 0]

            cache[key] = (policy, value)
            if len(cache) > max_cache_size:
                cache.popitem(last=False)

            return policy, value

        return model_policy_value

    def build_models(self):
        """
        Builds both checkpoint and best model
        May be overwritten later by loaded weights
        """
        self.checkpoint_model.build()
        self.best_model.build()

    def load_transposition_table(self):
        #TODO: Add this.  For now, just use empty table.

        warnings.warn("load_transposition_table is not properly implemented", stacklevel=2)

        self.prebuilt_transposition_table = {}

    def load_models(self):
        """ 
        Finds the checkpoint model and the best model in the given naming scheme 
        and loads those
        """
        import os

        # load checkpoint model
        
        for version in VERSIONS:
            model_files = [f for f in os.listdir('./save/') 
                                 if f.startswith("checkpoint_model_{}_gen".format(version))
                                 and f.endswith(".h5")]
            if model_files:
                # choose newest generation
                model_file = max(model_files, 
                                      key=lambda f: str_between(f, "_gen", ".h5"))
                path = "./save/" + model_file
                
                print("checkpoint model found:", "'" + path + "'")
                print("loading model ...")
                self.checkpoint_model.load_from_file(path)

                self.generation = int(str_between(path, "_gen", ".h5"))
                break

            else:
                print("no checkpoint model found with version {}".format(version))
        
        print("generation set to", self.generation)

        # load best model
        for version in VERSIONS:
            model_files = [f for f in os.listdir('./save/') 
                                 if f.startswith("model_{}_gen".format(version))
                                 and f.endswith(".h5")]
            if model_files:
                # choose newest generation
                model_file = max(model_files, 
                                      key=lambda f: (str_between(f, "_gen", ".h5")))
                path = "./save/" + model_file
                
                print("best model found:", "'" + path + "'")
                print("loading model ...")
                self.best_model.load_from_file(path)

                self.best_generation = int(str_between(path, "_gen", ".h5"))
                break

            else:
                print("no best model found with version {}".format(version)) 

        print("best generation:", self.best_generation)

    def save_checkpoint_model(self):
        file_name = "checkpoint_model_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name
        self.checkpoint_model.save_to_file(path)
        print("saved model checkpoint:", "'" + path + "'")

        self.checkpoint_training_distance_level = self.training_distance_level
        self.checkpoint_recent_wins = Counter()
        self.checkpoint_recent_games = Counter()
        # add a few free wins to speed up the convergence
        for dist in range(int(self.training_distance_level) + 1):
            self.checkpoint_recent_games[dist] += 1
            self.checkpoint_recent_wins[dist] += 1

    def save_and_set_best_model(self):
        file_name = "model_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name
        self.checkpoint_model.save_to_file(path)
        print("saved model:", "'" + path + "'")

        self.best_model.load_from_file(path)

        self.best_generation = self.generation
        self.training_distance_level = self.checkpoint_training_distance_level
        self.recent_wins = self.checkpoint_recent_wins
        self.recent_games = self.checkpoint_recent_games

    def train_model(self):
        import os
        import h5py

        inputs_list = []
        outputs_policy_list = []
        outputs_value_list = []

        counter = 0
        for version in VERSIONS:
            if counter > self.prev_generations_used_for_training:
                break

            data_files = [(str_between(f, "_gen", ".h5"), f)
                                for f in os.listdir('./save/') 
                                if f.startswith("data_{}_gen".format(version))
                                and f.endswith(".h5")]
            
            # go through in reverse order
            for gen, f in reversed(data_files):
                if counter > self.prev_generations_used_for_training:
                    break
                
                path = "./save/" + f

                print("loading data:", "'" + path + "'")

                with h5py.File(path, 'r') as hf:
                    inputs_list.append(hf['inputs'][:])
                    outputs_policy_list.append(hf['outputs_policy'][:])
                    outputs_value_list.append(hf['outputs_value'][:])

                counter += 1

        inputs_all = np.concatenate(inputs_list, axis=0)
        outputs_policy_all = np.concatenate(outputs_policy_list, axis=0)
        outputs_value_all = np.concatenate(outputs_value_list, axis=0)

        if self.validate_training_data:
            print("validating data...")
            self.checkpoint_model.validate_data(inputs_all, outputs_policy_all, outputs_value_all, gamma=self.decay)
            self.validate_training_data = False # just validate for first round
            print("data valid.")

        inputs_all, outputs_policy_all, outputs_value_all = \
            self.checkpoint_model.process_training_data(inputs_all, outputs_policy_all, outputs_value_all, augment=True)

        n = len(inputs_all)
        sample_size = int((n * self.training_sample_ratio) // 32 + 1) * 32 # roughly self.training_sample_ratio % of samples
        sample_idx = np.random.choice(n, size=sample_size)
        inputs = inputs_all[sample_idx]
        outputs_policy = outputs_policy_all[sample_idx]
        outputs_value = outputs_value_all[sample_idx]

        print("training...")
        self.checkpoint_model.train_on_data([inputs, outputs_policy, outputs_value])

    def reset_self_play(self):
        # Training parameters (dynamic)
        self.game_number = 0
        self.self_play_start = None # date and time (utc)
        self.self_play_end = None
        self.training_start = None
        self.training_end = None

        # Self play stats
        self.self_play_stats = defaultdict(list)
        self.game_stats = defaultdict(list)
        self.generation_stats = defaultdict(list)

        # Training data (one item per game based on randomly chosen game state)
        self.training_data_states = []
        self.training_data_policies = []
        self.training_data_values = []

        # set start time
        self.self_play_start = datetime.utcnow() # date and time (utc)

    def __remove_me__play_game(self, model, state=None, distance=None, evaluation_game=False):
        if distance is None:
            # choose distance
            lower_dist = int(self.training_distance_level)
            prob_of_increase = self.training_distance_level - lower_dist
            distance = lower_dist + np.random.choice(2, p=[1-prob_of_increase, prob_of_increase])
            
            lower_dist_win_rate = 0. if self.recent_games[lower_dist] == 0 else self.recent_wins[lower_dist] / self.recent_games[lower_dist]
            upper_dist_win_rate = 0. if self.recent_games[lower_dist+1] == 0 else self.recent_wins[lower_dist+1] / self.recent_games[lower_dist+1]
        
            print("(DB) distance:", distance, 
                  "(level: {:.2f} win rates: {}: {:.2f} {}: {:.2f})".format(self.training_distance_level, lower_dist, lower_dist_win_rate, lower_dist+1, upper_dist_win_rate))
        if state is None:
            state = State()
            while state.done(): 
                state.reset_and_randomize(distance)

        mcts = MCTSAgent(model.function, 
                         state, 
                         max_depth=self.max_depth, 
                         transposition_table=self.prebuilt_transposition_table.copy(),
                         c_puct = self.exploration,
                         gamma = self.decay)

        counter = 0
        win = True
        while not mcts.is_terminal():
            print("(DB) step:", counter)

            mcts.search(steps=self.max_steps)

            # find next state
            probs = mcts.action_probabilities(inv_temp = 10)
            action = np.argmax(probs)
            #action = np.random.choice(12, p=probs)

            shortest_path = mcts.stats('shortest_path')

            if not evaluation_game:
                # record stats
                self.self_play_stats['_game_id'].append(self.game_number)
                self.self_play_stats['_step_id'].append(counter)
                #self.self_play_stats['state']  # find a better representation of the state (that is easy to import)
                self.self_play_stats['shortest_path'].append(shortest_path)
                self.self_play_stats['action'].append(action)
                self.self_play_stats['value'].append(mcts.stats('value'))

                self.self_play_stats['prior'].append(mcts.stats('prior'))
                self.self_play_stats['prior_dirichlet'].append(mcts.stats('prior_dirichlet'))
                self.self_play_stats['visit_counts'].append(mcts.stats('visit_counts'))
                self.self_play_stats['total_action_values'].append(mcts.stats('total_action_values'))

                # training data (also recorded in stats)
                self.training_data_states.append(mcts.initial_node.state.input_array())
                
                policy = mcts.action_probabilities(inv_temp = 10)
                self.training_data_policies.append(policy)
                self.self_play_stats['updated_policy'].append(policy)
                
                self.training_data_values.append(0) # updated if game is success
                self.self_play_stats['updated_value'].append(0)

            # prepare for next state
            counter += 1 
            if shortest_path < 0:
                print("(DB) no path")
            if (counter > 1 and shortest_path < 0) or counter >= self.max_game_length:
                win = False
                break
            mcts.advance_to_action(action)
            

        # update training values based on game results
        if not evaluation_game:
            if win:
                value = 1
                for i in range(counter):
                    value *= self.decay
                    self.training_data_values[-(i+1)] = value
                    self.self_play_stats['updated_value'][-(i+1)] = value
        
            # record game stats
            self.game_stats['_game_id'].append(self.game_number)
            self.game_stats['distance_level'].append(self.training_distance_level)
            self.game_stats['training_distance'].append(distance)
            self.game_stats['max_game_length'].append(self.max_game_length)
            self.game_stats['win'].append(win)
            self.game_stats['total_steps'].append(counter if win else -1)

        # set up for next game
        self.game_number += 1
        if win:
            print("(DB)", "win")
        else:
            print("(DB)", "lose")

        if not evaluation_game:
            self.recent_wins[distance] += win
            self.recent_games[distance] += 1
            
            # update difficulty
            upper_dist = 0
            while True:
                upper_dist += 1
                if self.recent_wins[upper_dist] <= self.win_rate_target * self.recent_games[upper_dist]:
                    break
            if upper_dist <= self.min_distance:
                self.training_distance_level = float(self.min_distance)
            else:
                lower_dist = upper_dist - 1
                lower_dist_win_rate = 0. if self.recent_games[lower_dist] == 0 \
                                        else self.recent_wins[lower_dist] / self.recent_games[lower_dist]
                upper_dist_win_rate = 0. if self.recent_games[lower_dist+1] == 0 \
                                        else self.recent_wins[lower_dist+1] / self.recent_games[lower_dist+1]
                # notice that we won't divide by zero here since upper_dist_win_rate < lower_dist_win_rate
                self.training_distance_level = lower_dist + (lower_dist_win_rate - self.win_rate_target) / (lower_dist_win_rate - upper_dist_win_rate)

        return state, distance, win

    def save_training_stats(self):
        import pandas as pd

        file_name = "stats_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name

        # record time of end of self-play
        self.self_play_end = datetime.utcnow()

        # save generation_stats data
        self.generation_stats['_generation'].append(self.generation)
        self.generation_stats['best_model_generation'].append(self.best_generation)
        self.generation_stats['distance_level'].append(self.training_distance_level)
        self.generation_stats['memory_usage'].append(memory_used())
        self.generation_stats['version_history'].append(",".join(VERSIONS))
        self.generation_stats['self_play_start_datetime_utc'].append(str(self.self_play_start))
        self.generation_stats['self_play_end_datetime_utc'].append(str(self.self_play_end))
        self.generation_stats['self_play_time_sec'].append((self.self_play_end - self.self_play_start).total_seconds())
        
        generation_stats_df = pd.DataFrame(data=self.generation_stats)
        generation_stats_df.to_hdf(path, 'generation_stats', mode='a', format='fixed') #use mode='a' to avoid overwriting

        # save game_stats data
        game_stats_df = pd.DataFrame(data=self.game_stats)
        game_stats_df.to_hdf(path, 'game_stats', mode='a', format='fixed')
        
        # save self_play_stats data
        self_play_stats_df = pd.DataFrame(data=self.self_play_stats)
        self_play_stats_df.to_hdf(path, 'self_play_stats', mode='a', format='fixed') #use mode='a' to avoid overwriting

        print("saved stats:", "'" + path + "'")

    def __remove_me__process_training_data(self, inputs, policies, values):
        """
        Convert training data to arrays.  
        Augment with symmetric rotations.  
        Reshape to fit model input.
        """
        from batch_cube import position_permutations, color_permutations, action_permutations
        
        inputs = np.array(inputs).reshape((-1, 54, 6))
        sample_size = inputs.shape[0]

        # augement with all color rotations
        sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis, np.newaxis]
        pos_perm = position_permutations[:, np.newaxis, :, np.newaxis]
        col_perm = color_permutations[:, np.newaxis, np.newaxis, :]
        inputs = inputs[sample_idx, pos_perm, col_perm]
        inputs = inputs.reshape((-1, 54, 6))

        policies = np.array(policies)
        sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis]
        action_perm = action_permutations[:, np.newaxis, :]
        policies = policies[sample_idx, action_perm]
        policies = policies.reshape((-1, 12))
        
        values = np.array(values).reshape((-1, ))
        values = np.tile(values, 48)

        return inputs, policies, values

    def save_training_data(self):
        # save training_data
        import h5py

        file_name = "data_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name

        inputs, outputs_policy, outputs_value = \
            self.best_model.preprocess_training_data(self.training_data_states,
                                                  self.training_data_policies,
                                                  self.training_data_values)

        if self.validate_training_data:
            print("validating data...")
            self.best_model.validate_data(inputs, outputs_policy, outputs_value, gamma=self.decay)
            print("data valid.")

        with h5py.File(path, 'w') as hf:
            hf.create_dataset("inputs",  data=inputs)
            hf.create_dataset("outputs_policy",  data=outputs_policy)
            hf.create_dataset("outputs_value",  data=outputs_value)

        print("saved data:", "'" + path + "'")

    @staticmethod
    def random_state(distance):
        state = State()
        while state.done(): 
            state.reset_and_randomize(distance)
        return state

    @staticmethod
    def random_distance(distance_level):
        lower_dist = int(distance_level)
        prob_of_increase = distance_level - lower_dist
        distance = lower_dist + np.random.choice(2, p=[1-prob_of_increase, prob_of_increase]) #+ np.random.poisson(lam=1)
        return distance

    def update_win_and_level(self, distance, win, checkpoint=False):
        if checkpoint:
            training_distance_level = self.checkpoint_training_distance_level
            recent_wins = self.checkpoint_recent_wins
            recent_games = self.checkpoint_recent_games
        else:
            training_distance_level = self.training_distance_level
            recent_wins = self.recent_wins
            recent_games = self.recent_games

        # update wins/loses
        recent_wins[distance] += win
        recent_games[distance] += 1

        # update difficulty
        upper_dist = 0
        while True:
            upper_dist += 1
            if recent_wins[upper_dist] <= self.win_rate_target * recent_games[upper_dist]:
                break
        if upper_dist <= self.min_distance:
            training_distance_level = float(self.min_distance)
        else:
            lower_dist = upper_dist - 1
            lower_dist_win_rate = (.99 * self.win_rate_target) if recent_games[lower_dist] == 0 \
                                    else recent_wins[lower_dist] / recent_games[lower_dist]
            upper_dist_win_rate = (.99 * self.win_rate_target) if recent_games[lower_dist+1] == 0 \
                                    else recent_wins[lower_dist+1] / recent_games[lower_dist+1]
            # notice that we won't divide by zero here since upper_dist_win_rate < lower_dist_win_rate
            training_distance_level = lower_dist + (lower_dist_win_rate - self.win_rate_target) / (lower_dist_win_rate - upper_dist_win_rate)

        if checkpoint:
            self.checkpoint_training_distance_level = training_distance_level
        else:
            self.training_distance_level = training_distance_level

    def print_game_stats(self, game_results):
        game_id = game_results.game_id
        distance = game_results.distance
        level = game_results.distance_level
        win = game_results.win
        steps = game_results.game_stats['total_steps'][0]
        lost_way = game_results.self_play_stats['shortest_path'][0] < 0

        print("\nGame {}/{}".format(game_id, self.games_per_generation))
        print("distance: {} (level: {:.2f})".format(distance, level))
        if win:
            print("win ({}{} steps)".format(steps, "*" if lost_way else ""))
        else:
            print("loss")
        print()
        new_level = self.training_distance_level
        lower_dist = int(new_level)
        lower_dist_win_rate = float('nan') if self.recent_games[lower_dist] == 0 else self.recent_wins[lower_dist] / self.recent_games[lower_dist]
        upper_dist_win_rate = float('nan') if self.recent_games[lower_dist+1] == 0 else self.recent_wins[lower_dist+1] / self.recent_games[lower_dist+1]
        print("(DB) new level: {:.2f}, win rates: {}: {:.2f} {}: {:.2f}".format(new_level, lower_dist, lower_dist_win_rate, lower_dist+1, upper_dist_win_rate))
        print(end="", flush=True) # force stdout to flush (fixes buffering issues)

    def print_eval_game_stats(self, game_results1, game_results2, current_scores):
        game_id1 = game_results1.game_id
        game_id2 = game_results2.game_id
        distance1 = game_results1.distance
        distance2 = game_results2.distance
        level1 = game_results1.distance_level
        level2 = game_results2.distance_level
        win1 = game_results1.win
        win2 = game_results2.win
        steps1 = game_results1.game_stats['total_steps'][0]
        steps2 = game_results2.game_stats['total_steps'][0]
        lost_way1 = game_results1.self_play_stats['shortest_path'][0] < 0
        lost_way2 = game_results2.self_play_stats['shortest_path'][0] < 0
        assert game_id1 == game_id2
        assert distance1 == distance2
        print("\nEvaluation Game {}/{}".format(game_id1, self.games_per_evaluation))
        print("distance: {} (levels: {:.2f} {:.2f})".format(distance1, level1, level2))
        if win1:
            print("best model:       win ({}{} steps)".format(steps1, "*" if lost_way1 else ""))
        else:
            print("best model:       loss")
        if win2:
            print("checkpoint model: win ({}{} steps)".format(steps2, "*" if lost_way2 else ""))
        else:
            print("checkpoint model: loss")

        print()
        new_level = self.training_distance_level
        recent_games = self.recent_games
        recent_wins = self.recent_wins
        lower_dist = int(new_level)
        lower_dist_win_rate = float('nan') if recent_games[lower_dist] == 0 else recent_wins[lower_dist] / recent_games[lower_dist]
        upper_dist_win_rate = float('nan') if recent_games[lower_dist+1] == 0 else recent_wins[lower_dist+1] / recent_games[lower_dist+1]
        print("(DB) best model new level: {:.2f}, win rates: {}: {:.2f} {}: {:.2f}".format(new_level, lower_dist, lower_dist_win_rate, lower_dist+1, upper_dist_win_rate))
        
        new_level = self.checkpoint_training_distance_level
        recent_games = self.checkpoint_recent_games
        recent_wins = self.checkpoint_recent_wins
        lower_dist = int(new_level)
        lower_dist_win_rate = float('nan') if recent_games[lower_dist] == 0 else recent_wins[lower_dist] / recent_games[lower_dist]
        upper_dist_win_rate = float('nan') if recent_games[lower_dist+1] == 0 else recent_wins[lower_dist+1] / recent_games[lower_dist+1]
        print("(DB) checkpoint new level: {:.2f}, win rates: {}: {:.2f} {}: {:.2f}".format(new_level, lower_dist, lower_dist_win_rate, lower_dist+1, upper_dist_win_rate))
        print(end="", flush=True) # force stdout to flush (fixes buffering issues)

    def __remove_me__play_game(self, model, state, distance=None, evaluation_game=False):
        self_play_stats = defaultdict(list)
        game_stats = defaultdict(list)
        training_data_states = []
        training_data_policies = []
        training_data_values = []

        mcts = MCTSAgent(model.function, 
                         state, 
                         max_depth=self.max_depth, 
                         transposition_table=self.prebuilt_transposition_table.copy(),
                         c_puct = self.exploration,
                         gamma = self.decay)

        counter = 0
        win = True
        while not mcts.is_terminal():
            print("(DB) step:", counter)

            mcts.search(steps=self.max_steps)

            # find next state
            probs = mcts.action_probabilities(inv_temp = 10)
            action = np.argmax(probs)
            #action = np.random.choice(12, p=probs)

            shortest_path = mcts.stats('shortest_path')

            if not evaluation_game:
                # record stats
                self.self_play_stats['_game_id'].append(self.game_number)
                self.self_play_stats['_step_id'].append(counter)
                #self.self_play_stats['state']  # find a better representation of the state (that is easy to import)
                self.self_play_stats['shortest_path'].append(shortest_path)
                self.self_play_stats['action'].append(action)
                self.self_play_stats['value'].append(mcts.stats('value'))

                self.self_play_stats['prior'].append(mcts.stats('prior'))
                self.self_play_stats['prior_dirichlet'].append(mcts.stats('prior_dirichlet'))
                self.self_play_stats['visit_counts'].append(mcts.stats('visit_counts'))
                self.self_play_stats['total_action_values'].append(mcts.stats('total_action_values'))

                # training data (also recorded in stats)
                self.training_data_states.append(mcts.initial_node.state.input_array())
                
                policy = mcts.action_probabilities(inv_temp = 10)
                self.training_data_policies.append(policy)
                self.self_play_stats['updated_policy'].append(policy)
                
                self.training_data_values.append(0) # updated if game is success
                self.self_play_stats['updated_value'].append(0)

            # prepare for next state
            counter += 1 
            if shortest_path < 0:
                print("(DB) no path")
            if (counter > 1 and shortest_path < 0) or counter >= self.max_game_length:
                win = False
                break
            mcts.advance_to_action(action)
            

        # update training values based on game results
        if not evaluation_game:
            if win:
                value = 1
                for i in range(counter):
                    value *= self.decay
                    self.training_data_values[-(i+1)] = value
                    self.self_play_stats['updated_value'][-(i+1)] = value
        
            # record game stats
            self.game_stats['_game_id'].append(self.game_number)
            self.game_stats['distance_level'].append(self.training_distance_level)
            self.game_stats['training_distance'].append(distance)
            self.game_stats['max_game_length'].append(self.max_game_length)
            self.game_stats['win'].append(win)
            self.game_stats['total_steps'].append(counter if win else -1)

        data = None
        game_stats = None
        self_play_stats = None
        return win, data, game_stats, self_play_stats

    def state_generator(self, max_game_id, evaluation=False):
        while self.game_number < max_game_id:
            if evaluation:
                distance_level = max(self.training_distance_level, self.checkpoint_training_distance_level)
            else:
                distance_level = self.training_distance_level 
            distance = self.random_distance(distance_level)
            state = self.random_state(distance)

            print("(DB)", "starting game", self.game_number, "...")
            yield self.game_number, state, distance, distance_level

            self.game_number += 1

    def game_generator(self, model, state_generator, max_batch_size, return_in_order):
        """
        Send games to the batch game agent and retrieve the finished games.
        Yield the finished games in consecutive order of their id.
        """
        import heapq
        finished_games = [] # priority queue

        batch_game_agent = BatchGameAgent(model=model,
                                          max_steps=self.max_steps, 
                                          max_depth=self.max_depth,
                                          max_game_length=self.max_game_length, 
                                          transposition_table=self.prebuilt_transposition_table,
                                          decay=self.decay, 
                                          exploration=self.exploration) 

        # scale batch size up to make for better beginning determination of distance level
        # use batch size of 1 for first 16 games
        batch_size = 1
        cnt = 16

        # attach inital batch
        first_batch = list(itertools.islice(state_generator, batch_size))
        if not first_batch:
            return
        batch_game_agent.append_states(first_batch)
        next_game_id = first_batch[0][0] # game_id is first element

        # loop until all done
        while not batch_game_agent.is_empty():
            if self.multithreaded:
                batch_game_agent.run_one_step_with_threading()
            else:
                batch_game_agent.run_one_step()

            # collect all finished games
            for game_results in batch_game_agent.finished_game_results():
                heapq.heappush(finished_games, (game_results.game_id, game_results))

            # check if available slots
            if len(batch_game_agent.game_agents) < batch_size:
                
                # increment batch size
                cnt -= 1
                if cnt < 0:
                    batch_size = max_batch_size

            if return_in_order:
                # return those which are next in order
                if not finished_games or finished_games[0][1].game_id != next_game_id:
                    print("(DB)", "waiting on game", next_game_id, "(finished games:", ",".join(str(g[1].game_id) for g in finished_games), ") ...")

                while finished_games and finished_games[0][1].game_id == next_game_id:
                    yield heapq.heappop(finished_games)[1]
                    next_game_id += 1
            else:
                # return in order they are finished
                if not finished_games:
                    print("(DB) ...")

                while finished_games:
                    yield heapq.heappop(finished_games)[1]

            # fill up the batch (do after yields to ensure that self.training_distance_level is updated)
            available_slots = batch_size - len(batch_game_agent.game_agents)
            replacement_batch = itertools.islice(state_generator, available_slots)
            batch_game_agent.append_states(replacement_batch)

    def generate_data_self_play(self):
        # don't reset self_play since using the evaluation results to also get data
        #self.reset_self_play()

        for game_results in self.game_generator(self.best_model, self.state_generator(self.games_per_generation), max_batch_size=self.batch_size, return_in_order=False):
            # update data
            for k, v in game_results.self_play_stats.items():
                self.self_play_stats[k] += v
            for k, v in game_results.game_stats.items():
                self.game_stats[k] += v
            self.training_data_states += game_results.data_states
            self.training_data_policies += game_results.data_policies
            self.training_data_values += game_results.data_values
            
            # update win rates and level
            self.update_win_and_level(game_results.distance, game_results.win)

            # Print details
            self.print_game_stats(game_results)

    def evaluate_and_choose_best_model(self):
        self.reset_self_play()

        state_generator1, state_generator2 = itertools.tee(self.state_generator(self.games_per_evaluation, evaluation=True))

        best_model_wins = 0
        checkpoint_model_wins = 0
        ties = 0

        for game_results1, game_results2 \
            in zip(self.game_generator(self.best_model, state_generator1, max_batch_size=self.batch_size, return_in_order=True), 
                   self.game_generator(self.checkpoint_model, state_generator2, max_batch_size=self.batch_size, return_in_order=True)):

            if game_results1.win > game_results2.win:
                best_model_wins += 1
                game_results = game_results1
            elif game_results1.win < game_results2.win:
                checkpoint_model_wins += 1
                game_results = game_results2
            else:
                ties += 1
                game_results = game_results1

            # update data
            for k, v in game_results.self_play_stats.items():
                self.self_play_stats[k] += v
            for k, v in game_results.game_stats.items():
                self.game_stats[k] += v
            self.training_data_states += game_results.data_states
            self.training_data_policies += game_results.data_policies
            self.training_data_values += game_results.data_values

            # update win rates and level
            self.update_win_and_level(game_results1.distance, game_results1.win)
            self.update_win_and_level(game_results2.distance, game_results2.win, checkpoint=True)

            # Print details
            self.print_eval_game_stats(game_results1, game_results2, [best_model_wins, checkpoint_model_wins, ties])

        print("\nEvaluation results (win/lose/tie)")
        print("Best model      : {:2} / {:2} / {:2}".format(best_model_wins, checkpoint_model_wins, ties))
        print("Checkpoint model: {:2} / {:2} / {:2}".format(checkpoint_model_wins, best_model_wins, ties))
        
        if checkpoint_model_wins - best_model_wins > 5:
            print("\nCheckpoint model is better.")
            print("\nSave and set as best model...")
            self.save_and_set_best_model()
        else:
            print("\nCurrent best model is still the best.")

def main():
    agent = TrainingAgent()

    print("Build models...")
    agent.build_models()

    print("\nLoad pre-built transposition table...")
    agent.load_transposition_table()

    print("\nLoad models (if any)...")
    agent.load_models()
    
    print("\nBegin training loop...")
    agent.reset_self_play()

    while True:
        print("\nBegin self-play data generation...")
        agent.generate_data_self_play()
        break
        print("\nSave stats...")
        agent.save_training_stats()

        print("\nSave data...")
        agent.save_training_data()

        agent.generation += 1

        print("\nTrain model...")
        agent.train_model()

        print("\nSave model...")
        agent.save_checkpoint_model()   

        print("\nBegin evaluation...")
        agent.evaluate_and_choose_best_model()

if __name__ == '__main__':
    try:
        init_yappi()
        main()
    except KeyboardInterrupt:
        print("\nExiting the program...\nGood bye!")
    finally:
        pass
    
    