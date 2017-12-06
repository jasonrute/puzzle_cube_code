"""
End to end training of my neural network model.

The training routine has three key phases

- Evaluation through MCTS
- Data generation through MCTS
- Neural network training
"""
import numpy as np
from collections import defaultdict, deque, Counter
import warnings
import os, psutil # useful for memory management
from datetime import datetime

from mcts_nn_cube import State, MCTSAgent

# this keeps track of the training runs, including the older versions that we are extending
VERSIONS = ["v0.7.test9.2", "v0.7.test9.1", "v0.7.test9"]

# memory management
MY_PROCESS = psutil.Process(os.getpid())
def memory_used():
    return MY_PROCESS.memory_info().rss

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

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

class TrainingAgent():
    """
    This agent handles all the details of the training.
    """
    def __init__(self):
        # Model (NN) parameters (fixed)
        self.state_dim = (6*54, )
        self.action_count = 12
        self.checkpoint_model = None # model used for training (built later)
        self.best_model = None # model used for data generation (built later)
        self.checkpoint_policy_value = None # function used for training (built later)
        self.best_policy_value = None # function used for data generation (built later)
        self.learning_rate = .001

        # MCTS parameters (fixed)
        self.max_depth = 900
        self.max_steps = 1600
        self.use_prebuilt_transposition_table = False
        self.decay = 0.95 # gamma
        self.exploration = 1. #c_puct
        self.prebuilt_transposition_table = None # built later

        # Training parameters (fixed)
        self.games_per_generation = 100
        self.starting_distance = 1
        self.min_distance = 1
        self.win_rate_target = .5
        self.max_game_length = 100
        self.prev_generations_used_for_training = 10
        self.training_sample_size = 2024 * 64
        self.games_per_evaluation = 100

        # Training parameters preserved between generations
        self.training_distance_level = float(self.starting_distance)
        self.recent_wins = Counter()
        self.recent_games = Counter()

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

    def starting_model(self):
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
            conv = conv_block(block, filter_size=32)
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
            conv = conv_block(block, filter_size=32)
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
        block = conv_block(state_input, filter_size=32)

        # 2 residuals
        block = residual_block(block, filter_size=32)
        block = residual_block(block, filter_size=32)

        # policy head
        policy_output = policy_block(block, filter_size=32, hidden_size=32)

        # value head
        value_output = value_block(block, filter_size=32, hidden_size=32)

        # combine
        model = Model(inputs=state_input, outputs=[policy_output, value_output])
        model.compile(loss={'policy_output': categorical_crossentropy, 
                            'value_output': 'mse'},
                      loss_weights={'policy_output': 1., 'value_output': 1.},
                      optimizer=Adam(lr=self.learning_rate))

        return model

    def build_model_policy_value(self, model, max_cache_size=100000):
        from collections import OrderedDict
        cache = OrderedDict()
        from keras import backend as K
        get_output = K.function([model.input, K.learning_phase()], [model.output[0], model.output[1]])
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
        self.checkpoint_model = self.starting_model()
        self.best_model = self.starting_model()

        self.checkpoint_policy_value = self.build_model_policy_value(self.checkpoint_model)
        self.best_policy_value = self.build_model_policy_value(self.best_model)

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
                self.checkpoint_model.load_weights(path)
                self.checkpoint_policy_value = self.build_model_policy_value(self.checkpoint_model)

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
                self.best_model.load_weights(path)
                self.best_policy_value = self.build_model_policy_value(self.best_model)

                self.best_generation = int(str_between(path, "_gen", ".h5"))
                break

            else:
                print("no best model found with version {}".format(version)) 

        print("best generation:", self.best_generation)

    def save_checkpoint_model(self):
        file_name = "checkpoint_model_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name
        self.checkpoint_model.save_weights(path)
        print("saved model checkpoint:", "'" + path + "'")

    def save_and_set_best_model(self):
        file_name = "model_{}_gen{:03}.h5".format(VERSIONS[0], self.generation)
        path = "./save/" + file_name
        self.checkpoint_model.save_weights(path)
        print("saved model:", "'" + path + "'")

        self.best_model.load_weights(path)
        self.best_policy_value = self.build_model_policy_value(self.best_model)

        self.best_generation = self.generation
        self.recent_wins = Counter()
        self.recent_games = Counter()

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

        n = len(inputs_all)
        sample_idx = np.random.choice(n, size=self.training_sample_size)
        inputs = inputs_all[sample_idx]
        outputs_policy = outputs_policy_all[sample_idx]
        outputs_value = outputs_value_all[sample_idx]

        self.checkpoint_model.fit(x=inputs, 
                                  y={'policy_output': outputs_policy, 'value_output': outputs_value}, 
                                  epochs=1, verbose=0)

        self.checkpoint_policy_value = self.build_model_policy_value(self.checkpoint_model)

    def batch_test(self):
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

        n = len(inputs_all)
        sample_idx = np.random.choice(n, size=2**10)
        inputs = inputs_all[sample_idx]
        outputs_policy = outputs_policy_all[sample_idx]
        outputs_value = outputs_value_all[sample_idx]

        import time
        from keras import backend as K
        get_output = K.function([self.best_model.input, K.learning_phase()], [self.best_model.output[0], self.best_model.output[1]])
        for i in range(11):
            my_inputs = inputs.copy().reshape((-1, 2**i, 54, 6))
            print("test batch size:", 2**i)
            t1 = time.time()
            for batch in my_inputs:
                get_output([batch, 0])
            print("time:", time.time() - t1)
            t1 = time.time()
            for batch in my_inputs:
                self.best_model.predict(batch)
            print("time:", time.time() - t1)
            t1 = time.time()
            for batch in my_inputs:
                self.best_model.predict(batch, batch_size=2**i)
            print("time:", time.time() - t1)



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

    def play_game(self, model_policy_value, state=None, distance=None, evaluation_game=False):
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

        mcts = MCTSAgent(model_policy_value, 
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
            if shortest_path < 0 or counter >= self.max_game_length:
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
                # notice that we won't divide by zero hear since upper_dist_win_rate < lower_dist_win_rate
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

    def process_training_data(self, inputs, policies, values):
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
            self.process_training_data(self.training_data_states,
                                       self.training_data_policies,
                                       self.training_data_values)

        with h5py.File(path, 'w') as hf:
            hf.create_dataset("inputs",  data=inputs)
            hf.create_dataset("outputs_policy",  data=outputs_policy)
            hf.create_dataset("outputs_value",  data=outputs_value)

        print("saved data:", "'" + path + "'")

    def evaluate_model(self):
        warnings.warn("evaluate_model is not implemented", stacklevel=2)

def main():
    agent = TrainingAgent()

    print("Build models...")
    agent.build_models()

    print("\nLoad pre-built transposition table...")
    agent.load_transposition_table()

    print("\nLoad models (if any)...")
    agent.load_models()
    
    print("\nBegin training loop...")
    while True:
        print("\nBegin self-play data generation...")
        agent.reset_self_play()

        agent.batch_test()
        break

        for game in range(agent.games_per_generation):
            print("\nGame {}/{}".format(game, agent.games_per_generation))
            agent.play_game(agent.best_policy_value)
        
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
        agent.reset_self_play()

        best_model_wins = 0
        checkpoint_model_wins = 0
        ties = 0

        for game in range(agent.games_per_evaluation):
            print("\nEvaluation Game {}/{}".format(game, agent.games_per_evaluation))
            print("\nBest model")
            state, distance, win1 = agent.play_game(agent.best_policy_value, state=None, distance=None, evaluation_game=True)

            print("\nCheckpoint model")
            _, _, win2 = agent.play_game(agent.checkpoint_policy_value, state=state, distance=distance, evaluation_game=True)
            
            if win1 > win2:
                best_model_wins += 1
            elif win1 < win2:
                checkpoint_model_wins += 1
            else:
                ties += 1

        print("\nEvaluation results (win/lose/tie)")
        print("Best model      : {:2} / {:2} / {:2}".format(best_model_wins, checkpoint_model_wins, ties))
        print("Checkpoint model: {:2} / {:2} / {:2}".format(checkpoint_model_wins, best_model_wins, ties))
        
        if checkpoint_model_wins - best_model_wins > 5:
            print("\nCheckpoint model is better.")
            print("\nSave and set as best model...")
            agent.save_and_set_best_model()
        else:
            print("\nCurrent best model is still the best.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting the program...\nGood bye!")
    
    