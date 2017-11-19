"""
The various models.  These all have a common API (except for __init__ which may have extra
parameters) and are basically wrappers around the Keras models.
"""
from collections import OrderedDict
import numpy as np
import time
from batch_cube import position_permutations, color_permutations, action_permutations

def random_rotation_wrapper(model_policy_value):
    def rotationally_randomized_policy_value(input_array):
        # rotate colors
        input_array = input_array.reshape((54, 6))
        rotation_id = np.random.choice(48)
        pos_perm = position_permutations[rotation_id][:,np.newaxis]
        col_perm = color_permutations[rotation_id][np.newaxis]
        input_array = input_array[pos_perm, col_perm]

        policy, value = model_policy_value(input_array)

        return policy[opp_action_permutations[rotation_id]], value

    return rotationally_randomized_policy_value

def randomize_input(input_array):
    pos_perm = position_permutations[rotation_id][:,np.newaxis]
    col_perm = color_permutations[rotation_id][np.newaxis]
    input_array = input_array[pos_perm, col_perm]

    return input_array

def derandomize_policy(policy):
    return policy[opp_action_permutations[rotation_id]]

def augment_data(inputs, policies, values):
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

class ConvModel(): 
    """
    A residual 2D-convolutional model.
    """   

    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False):
        self.learning_rate = .001
        self._model = None # Built and/or loaded later
        self._run_count = 0 # used for measuring computation timing
        self._time_sum = 0 # used for measuring computation timing
        self._cache = None
        self._get_output = None
        self.use_cache = use_cache
        self.rotationally_randomize = rotationally_randomize
        self.max_cache_size = max_cache_size

    def build(self):
        """
        Build a new neural network using the below architecture
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
        model = Model(inputs=state_input, outputs=[policy_output, value_output])
        model.compile(loss={'policy_output': categorical_crossentropy, 
                            'value_output': 'mse'},
                      loss_weights={'policy_output': 1., 'value_output': 1.},
                      optimizer=Adam(lr=self.learning_rate))

        self._model = model
        self._rebuild_function()

    @staticmethod
    def process_single_input(input_array):
        input_array = input_array.reshape((-1, 54, 6))
        input_array = np.rollaxis(input_array, 2, 1).reshape(-1, 6*6, 3, 3)
        return input_array

    def _rebuild_function(self):
        """
        Rebuilds the function associated with this network.  This is called whenever
        the network is changed.
        """
        from keras import backend as K
        self._cache = OrderedDict()
        self._get_output = K.function([self._model.input, K.learning_phase()], [self._model.output[0], self._model.output[1]])

    def _raw_function(self, input_array):
        t1 = time.time()
        #return self._model.predict(input_array)
        out = self._get_output([input_array, 0])
        self._run_count += 1
        self._time_sum = time.time() - t1 
        return out

    def function(self, input_array):
        """
        The function which computes the output to the array
        """ 
        if self.rotationally_randomize:
            rotation_id = np.random.choice(48)
            randomize_input(input_array, rotation_id)

        if self.use_cache:
            key = input_array.tobytes()
            if key in self._cache:
                self._cache.move_to_end(key, last=True)
                return self._cache[key]
        
        input_array = self.process_single_input(input_array)
        policy, value = self._raw_function(input_array)
        
        policy = policy.reshape((12,))
        value = value[0, 0]

        if self.use_cache:
            self._cache[key] = (policy, value)
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)
        
        if self.rotationally_randomize:
            policy = derandomize_policy(policy, rotation_id)

        return policy, value

    def load_from_file(self, path):
        self._model.load_weights(path)
        self._rebuild_function()

    def save_to_file(self, path):
        self._model.save_weights(path)
    
    def train_on_data(self, data):
        """
        data: list of inputs, policies, values as arrays (assume already preprocessed for training)
        """

        inputs, outputs_policy, outputs_value = data

        self._model.fit(x=inputs, 
                                  y={'policy_output': outputs_policy, 'value_output': outputs_value}, 
                                  epochs=1, verbose=0)
        self._rebuild_function()

    @staticmethod
    def augment_data(inputs, policies, values):
        """
        Augment data with all 48 color rotations
        """
        from batch_cube import position_permutations, color_permutations, action_permutations

        inputs = np.array(inputs).reshape((-1, 54, 6))
        sample_size = inputs.shape[0]

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

    @staticmethod
    def validate_data(inputs, policies, values, gamma=.95):
        """
        Validate the input, policy, value data to make sure it is of good quality.
        It must be in order and not shuffled.
        """
        from batch_cube import BatchCube
        import math

        next_state = None
        next_value = None

        for state, policy, value in zip(inputs, policies, values):
            cube = BatchCube()
            cube.load_bit_array(state)

            if next_state is not None:
                assert next_state.shape == state.shape
                assert np.array_equal(next_state, state), "\nstate:\n" + str(state) + "\nnext_state:\n" + str(next_state)
            if next_value is not None:
                assert round(math.log(next_value, .95)) == round(math.log(value, .95)), "next_value:" + str(next_value) + "   value:" + str(value)

            action = np.argmax(policy)
            cube.step([action])

            if value == 0 or value == gamma:
                next_value = None
                next_state = None
            else:
                next_value = value / gamma
                next_state = cube.bit_array().reshape((54, 6))

    @staticmethod
    def preprocess_training_data(inputs, policies, values):
        """
        Convert training data to arrays in preparation for saving.
        
        Don't augment, since this takes up too much space and doesn't cost much in time to
        do it later.

        Also keep the inputs shape as (-1, 54, 6) so that other networks can use the same
        data.
        """
        
        # convert to arrays
        inputs = np.array(inputs).reshape((-1, 54, 6))
        policies = np.array(policies)
        values = np.array(values).reshape((-1, ))

        return inputs, policies, values

    @staticmethod
    def process_training_data(inputs, policies, values, augment=True):
        """
        Convert training data to arrays.  
        Augment data
        Reshape to fit model input.
        """
        
        # augment with all 48 color rotations
        if augment:
            inputs, policies, values = augment_data(inputs, policies, values)

        # process arrays now to save time during training
        inputs = inputs.reshape((-1, 54, 6))
        inputs = np.rollaxis(inputs, 2, 1).reshape(-1, 6*6, 3, 3)

        return inputs, policies, values


