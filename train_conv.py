"""
End to end training of my neural network model.

The training routine has three key phases

- Evaluation through MCTS
- Data generation through MCTS
- Neural network training
"""
import numpy as np
from collections import defaultdict
import warnings

from mcts_nn_cube import State, MCTSAgent

# this keeps track of the training runs, including the older versions that we are extending
VERSIONS = ["v0.3"] 

def str_between(s, start, end):
    return (s.split(start))[1].split(end)[0]

class TrainingAgent():
    """
    This agent handles all the details of the training.
    """
    def __init__(self):
        # Model (NN) parameters (fixed)
        self.state_dim = (6*54, )
        self.action_count = 12
        self.model = None # built later

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
        self.min_distance = 6
        self.win_rate_upper = .85
        self.win_rate_lower = .75
        self.max_game_length = 100

        # Training parameters (dynamic)
        self.training_distance = self.starting_distance
        self.total_wins = 0
        self.game_number = 0

        # Evaluation parameters (dynamic)
        self.generation = 0
        self.score = 0
        self.old_model = 0
        self.best_score = 0

        # Self play stats
        # These are functionally data tables implemented as a dictionary of lists
        # The keys are the column names.  This makes it easy to change the stats I am recording.
        self.self_play_stats = defaultdict(list)
        self.game_stats = defaultdict(list)
        self.training_stats = defaultdict(list)

        # Training data
        self.states = []
        self.policies = []
        self.values = []



    def build_model(self):
        """
        Build a neural network model
        """
        import numpy as np
        from keras.models import Model
        from keras.layers import Conv2D, Input, BatchNormalization, Dense, Flatten, Activation, add
        from keras.optimizers import Adam
        from keras.losses import categorical_crossentropy
        from keras.regularizers import l2

        state_input = Input(shape=(6 * 6, 3, 3), name='state_input')
        
        conv = Conv2D(64, kernel_size=3, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(state_input)
        batch = BatchNormalization(axis=1)(conv)
        end_of_block = Activation('relu')(batch)

        # residual block
        conv = Conv2D(64, kernel_size=3, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(end_of_block)
        batch = BatchNormalization(axis=1)(conv)
        relu = Activation('relu')(batch)
        conv = Conv2D(64, kernel_size=3, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(relu)
        batch = BatchNormalization(axis=1)(conv)
        conn = add([batch, end_of_block])
        end_of_block = Activation('relu')(conn)

        # residual block
        conv = Conv2D(64, kernel_size=3, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(end_of_block)
        batch = BatchNormalization(axis=1)(conv)
        relu = Activation('relu')(batch)
        conv = Conv2D(64, kernel_size=3, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(relu)
        batch = BatchNormalization(axis=1)(conv)
        conn = add([batch, end_of_block])
        end_of_block = Activation('relu')(conn)

        # policy head
        conv = Conv2D(64, kernel_size=1, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(end_of_block)
        batch = BatchNormalization(axis=1)(conv)
        relu = Activation('relu')(batch)
        flat = Flatten()(relu)
        hidden = Dense(64, activation='relu',
                             kernel_regularizer=l2(0.001), 
                             bias_regularizer=l2(0.001))(flat)
        policy_output = Dense(12, activation='softmax',
                                  kernel_regularizer=l2(0.001), 
                                  bias_regularizer=l2(0.001),
                                  name='policy_output')(hidden)

        # value head
        conv = Conv2D(64, kernel_size=1, 
                          strides=(1, 1), 
                          padding='same', 
                          data_format="channels_first",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(end_of_block)
        batch = BatchNormalization(axis=1)(conv)
        relu = Activation('relu')(batch)
        flat = Flatten()(relu)
        hidden = Dense(64, activation='relu',
                             kernel_regularizer=l2(0.001), 
                             bias_regularizer=l2(0.001))(flat)
        value_output = Dense(1, activation='sigmoid',
                                  kernel_regularizer=l2(0.001), 
                                  bias_regularizer=l2(0.001),
                                  name='value_output')(hidden)

        # combine
        self.model = Model(inputs=state_input, outputs=[policy_output, value_output])
        self.model.compile(loss={'policy_output': categorical_crossentropy, 
                                 'value_output': 'mse'},
                           loss_weights={'policy_output': 1., 'value_output': 1.},
                           optimizer=Adam(lr=0.001))

    def model_value_policy(self, input_array):
        input_array = input_array.reshape((-1, 54, 6))
        input_array = np.rollaxis(input_array, 2, 1).reshape(-1, 6*6, 3, 3)
        policy, value = self.model.predict(input_array)
        policy = policy.reshape((self.action_count,))
        value = value[0, 0]
        return policy, value

    def load_transposition_table(self):
        #TODO: Add this.  For now, just use empty table.

        warnings.warn("load_transposition_table is not properly implemented", stacklevel=2)

        self.prebuilt_transposition_table = {}

    def load_best_model(self):
        """ Finds the best model in the given naming scheme and loads that one """
        import os

        for version in VERSIONS:
            model_files = [f for f in os.listdir('./save/') 
                                 if f.startswith("model_{}_gen".format(version))
                                 and f.endswith(".h5")]
            if model_files:
                best_model_file = max(model_files, 
                                      key=lambda f: (str_between(f, "_score", ".h5"), 
                                                     str_between(f, "_gen", "_score")))
                path = "./save/" + best_model_file
                
                print("best model found:", "'" + path + "'")
                print("loading model ...")
                self.model.load_weights(path)
                
                self.generation = int(str_between(path, "_gen", "_score"))
                break

            else:
                print("no model found with version {}".format(version))
           
        print("generation set to", self.generation)

    def save_model(self):
        file_name = "model_{}_gen{:03}_score{:02}.h5".format(VERSION[0], self.generation, self.score)
        path = "./save/" + file_name
        self.model.save_weights(path)
        print("saved model:", "'" + path + "'")

    def train_model(self):
        inputs = np.array(self.states).reshape((-1, 54, 6))
        inputs = np.rollaxis(inputs, 2, 1).reshape(-1, 6*6, 3, 3)
        outputs_policy = np.array(self.policies)
        outputs_value = np.array(self.values)

        self.model.fit(x=inputs, 
                       y={'policy_output': outputs_policy, 'value_output': outputs_value}, 
                       epochs=1, verbose=0)

        self.generation += 1

    def reset_self_play(self):
        # Training parameters (dynamic)
        self.training_distance = self.starting_distance
        self.total_wins = 0
        self.game_number = 0

        # Self play stats
        self.self_play_stats = defaultdict(list)
        self.game_stats = defaultdict(list)

        # Training data (one item per game based on randomly chosen game state)
        self.states = []
        self.policies = []
        self.values = []

    def play_game(self):
        state = State()
        while state.done(): 
            state.reset_and_randomize(self.training_distance)

        mcts = MCTSAgent(self.model_value_policy, 
                         state, 
                         max_depth=self.max_depth, 
                         transposition_table=self.prebuilt_transposition_table.copy(),
                         c_puct = self.exploration,
                         gamma = self.decay)

        counter = 0
        win = True
        lost_path = False
        while not mcts.is_terminal():
            print("(DB) step:", counter, "training dist:", self.training_distance)
            if counter >= self.max_game_length:
                win = False
                break

            if lost_path:
                win = False
                break

            mcts.search(steps=self.max_steps)

            # find next state
            probs = mcts.action_probabilities(inv_temp = 1)
            action = np.argmax(probs)
            #action = np.random.choice(12, p=probs)

            # record stats
            self.self_play_stats['_game_id'].append(self.game_number)
            self.self_play_stats['_step_id'].append(counter)
            self.self_play_stats['state'].append(state.input_array())
            
            self.self_play_stats['shortest_path'].append(mcts.stats('shortest_path'))
            self.self_play_stats['value'].append(mcts.stats('value'))
            self.self_play_stats['prior'].append(mcts.stats('prior'))
            self.self_play_stats['prior_dirichlet'].append(mcts.stats('prior_dirichlet'))
            self.self_play_stats['visit_counts'].append(mcts.stats('visit_counts'))
            self.self_play_stats['total_action_values'].append(mcts.stats('total_action_values'))
            
            self.self_play_stats['policy'].append(mcts.action_probabilities(inv_temp = 1))
            self.self_play_stats['action'].append(action)
            
            # prepare for next state
            if mcts.stats('shortest_path') < 0:
                lost_path = True
            mcts.advance_to_action(action)
            counter += 1 

        # pick random step for training state
        i = np.random.choice(counter) + 1
        self.states.append(self.self_play_stats['state'][-i])
        self.policies.append(self.self_play_stats['policy'][-i])
        self.values.append(self.decay**i if win else 0)
        
        # record game stats
        self.game_stats['_game_id'].append(self.game_number-1)
        self.game_stats['training_distance'].append(self.training_distance)
        self.game_stats['max_game_length'].append(self.max_game_length)
        self.game_stats['win'].append(win)
        self.game_stats['total_steps'].append(counter if win else -1)
        self.game_stats['sample_step_id'].append(counter - i)

        # set up for next game
        self.game_number += 1
        self.total_wins += win 
        print("(DB) win:", win, "total_wins:", self.total_wins, 
              "upper:", self.win_rate_upper * self.game_number, 
              "lower:", self.win_rate_lower * self.game_number)
        if self.total_wins > self.win_rate_upper * self.game_number:
            # Too many wins, make it harder
            self.training_distance += 1
        elif self.total_wins < self.win_rate_lower * self.game_number:
            # Too few wins, make it easier
            if self.training_distance > self.min_distance:
                self.training_distance -= 1

    def save_training_stats(self):
        import pandas as pd

        file_name = "stats_{}_gen{:03}_score{:02}.h5".format(VERSION[0], self.generation, self.score)
        path = "./save/" + file_name

        # save game_stats data
        game_stats_df = pd.DataFrame(data=self.game_stats)
        game_stats_df.to_hdf(path, 'game_stats', mode='a', format='fixed')
        
        # save self_play_stats data
        self_play_stats_df = pd.DataFrame(data=self.self_play_stats)
        self_play_stats_df.to_hdf(path, 'self_play_stats', mode='a', format='fixed') #use mode='a' to avoid overwriting

        print("saved stats:", "'" + path + "'")

    def evaluate_model(self):
        warnings.warn("evaluate_model is not implemented", stacklevel=2)

def main():
    agent = TrainingAgent()

    print("Build model...")
    agent.build_model()

    print("\nLoad pre-built transposition table...")
    agent.load_transposition_table()

    print("\nLoad best model (if any)...")
    agent.load_best_model()

    print("\nBegin training loop...")
    while True:
        print("\nBegin self-play data generation...")
        agent.reset_self_play()

        for game in range(agent.games_per_generation):
            print("\nGame {}/{}".format(game, agent.games_per_generation))
            agent.play_game()

        print("\nTrain model...")
        agent.train_model()

        print("\nEvaluate model...")
        agent.evaluate_model()

        print("\nSave stats...")
        agent.save_training_stats()

        print("\nSave model...")
        agent.save_model()     


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting the program...\nGood bye!")
    
    