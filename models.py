"""
The various models.  These all have a common API (except for __init__ which may have extra
parameters) and are basically wrappers around the Keras models.
"""
from collections import OrderedDict
import numpy as np
import time
from batch_cube import BatchCube, position_permutations, color_permutations, action_permutations, opp_action_permutations
import warnings
import threading, queue


def randomize_input(input_array, rotation_id):
    """
    Randomizes the input, assuming the input has shape (-1, 54, 6)
    """
    pos_perm = position_permutations[rotation_id][np.newaxis, :, np.newaxis]
    col_perm = color_permutations[rotation_id][np.newaxis, np.newaxis]
    input_array = input_array[:, pos_perm, col_perm]

    return input_array

def derandomize_policy(policy, rotation_id):
    """
    Randomizes the policy, assuming the policy has shape (12, )
    """
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

class Task:
    def __init__(self):
        self.lock = threading.Condition()
        self.input = None
        self.output = None
        self.kill_thread = False

class BaseModel(): 
    """
    The Base Class for my models.  Assuming Keras/Tensorflow backend and
    that the input is a bit array representing a single cube (no history).
    """   
    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False, history=1):
        self.learning_rate = .001
        self._model = None # Built and/or loaded later
        self._run_count = 0 # used for measuring computation timing
        self._time_sum = 0 # used for measuring computation timing
        self._cache = None
        self._get_output = None
        self.use_cache = use_cache
        self.rotationally_randomize = rotationally_randomize
        self.max_cache_size = max_cache_size
        self.history = history

        # multithreading, batch evaluation support
        self.multithreaded = False
        self.ideal_batch_size = 128
        self._lock = threading.RLock()
        self._max_batch_size = 1
        self._queue = queue.Queue()
        self._worker_thread = None

        # to reimplement for each model.  Leave off the first dimension.
        self.input_shape = (54, self.history * 6)

    def set_max_batch_size(self, max_batch_size):
        with self._lock:
            self._max_batch_size = max_batch_size
        
        # put a dummy task on the queue to make sure the worker notices the update to the max_batch_size
        dummy_task = Task()
        with dummy_task.lock:
            self._queue.put(dummy_task)

    def get_max_batch_size(self):
        with self._lock:
            return self._max_batch_size

    def _build(self, model):
        """
        The final part of each build method.
        """
        self._model = model
        self._rebuild_function()

    def build(self):
        """
        Build a new neural network using the below architecture
        """
        warnings.warn("'BaseModel.build' should not be used.  The 'build' method should be reimplemented", stacklevel=2)

        model = None 
        self._build(self, model)

    def process_single_input(self, input_array):
        warnings.warn("'BaseModel.process_single_input' should not be used.  The 'process_single_input' method should be reimplemented", stacklevel=2)
        input_array = input_array.reshape((self.history, 54, 6))
        if self.history > 1:
            input_array = np.rollaxis(input_array,  1, 0)
            input_array = input_array.reshape((1, 54, self.history * 6))
        return input_array

    def _rebuild_function(self):
        """
        Rebuilds the function associated with this network.  This is called whenever
        the network is changed.
        """
        from keras import backend as K
        self._cache = OrderedDict()

        # run model once to make sure it loads correctly (needed for K.function to work on new models)
        trivial_input = np.zeros((1, ) + self.input_shape)
        self._model.predict(trivial_input)

        self._get_output = K.function([self._model.input, K.learning_phase()], [self._model.output[0], self._model.output[1]])
        
        if self.multithreaded:
            if self._worker_thread is not None:
                self.stop_worker_thread()
            self.start_worker_thread()

    def _raw_function(self, input_array):
        t1 = time.time()
        #return self._model.predict(input_array)
        out = self._get_output([input_array, 0])
        self._run_count += 1
        self._time_sum = time.time() - t1 
        return out

    def _raw_function_worker(self):
        import numpy as np
        task_list = []

        while True:
            # retrieve items from the queue
            task = self._queue.get()
            with task.lock:
                if task.kill_thread:
                    task.lock.notify()
                    return
                
                if task.input is not None: #ignore other tasks as dummy tasks
                    task_list.append(task)

            if task_list and len(task_list) >= min(self.ideal_batch_size, self.get_max_batch_size()):
                array = np.array([task.input.squeeze(axis=0) for task in task_list])
                policies, values = self._get_output([array, 0])

                for p, v, task in zip(policies, values, task_list):
                    with task.lock:
                        task.output = [p[np.newaxis], v[np.newaxis]]
                        task.lock.notify() # mark as being complete

                task_list = []

    def start_worker_thread(self):
        self._worker_thread = threading.Thread(target=self._raw_function_worker, args=())
        self._worker_thread.daemon = True
        self._worker_thread.start()

    def stop_worker_thread(self):
        poison_pill = Task()
        with poison_pill.lock:
            poison_pill.kill_thread = True
            self._queue.put(poison_pill) # put task on queue to be processed
            poison_pill.lock.wait() # wait for poison pill to be processed
        self._worker_thread.join() # wait until thread finishes
        self._worker_thread = None

    def _raw_function_pass_to_worker(self, input_array):
        task = Task()

        # put the value on the queue to be processed
        with task.lock:
            task.input = input_array
            self._queue.put(task) # put task on queue to be processed
            task.lock.wait() # wait until task is processed
            return task.output # return output

    def _inner_function(self, input_array):
        """
        The function which computes the output to the array.
        Assume input_array has shape (-1, 56, 4) where -1 represents the history.
        """ 
        if self.use_cache:
            key = input_array.tobytes()
            if key in self._cache:
                self._cache.move_to_end(key, last=True)
                return self._cache[key]
        
        input_array = self.process_single_input(input_array)
        if self.multithreaded:
            policy, value = self._raw_function_pass_to_worker(input_array)
        else:
            policy, value = self._raw_function(input_array)
        
        policy = policy.reshape((12,))
        value = value[0, 0]

        if self.use_cache:
            self._cache[key] = (policy, value)
            if len(self._cache) > self.max_cache_size:
                self._cache.popitem(last=False)

        return policy, value

    def function(self, input_array):
        """
        The function which computes the output to the array.
        If self.rotationally_randomize is true, will first randomly rotate input
        and (un-)rotate corresponding policy output.
        Assume input_array has shape (-1, 56, 4) where -1 represents the history.
        """ 
        if self.rotationally_randomize:
            rotation_id = np.random.choice(48)
            input_array = randomize_input(input_array, rotation_id)

        policy, value = self._inner_function(input_array)

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
        print("AAA done training")
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

        Also keep the inputs shape as (-1, 54, 6) so that other models can use the same
        data.  Similarly, assume the inputs are stored only with the last state.
        """
        
        # convert to arrays
        inputs = np.array(inputs).reshape((-1, 54, 6))
        policies = np.array(policies)
        values = np.array(values).reshape((-1, ))

        return inputs, policies, values

    def process_training_data(self, inputs, policies, values, augment=True):
        """
        Convert training data to arrays.  
        Augment data
        Reshape to fit model input.
        """
        warnings.warn("'BaseModel.process_training_data' should not be used.  The 'process_single_input' method should be reimplemented", stacklevel=2)
        # augment with all 48 color rotations
        if augment:
            inputs, policies, values = augment_data(inputs, policies, values)

        # process arrays now to save time during training
        if self.history == 1:
            inputs = inputs.reshape((-1, 54, 6))
        else:
            # use that the inputs are in order to attach the history
            # use the policy/input match to determine when we reached a new game
            next_cube = None
            input_array_with_history = None
            input_list = []
            for state, policy in zip(inputs, policies):
                cube = BatchCube()
                cube.load_bit_array(state)
                
                if next_cube is None or cube != next_cube:
                    # blank history
                    input_array_history = np.zeros((self.history-1, 54, 6), dtype=bool)
                else:
                    input_array_history = input_array_with_history[:-1]
                
                input_array_state = state.reshape((1, 54, 6))
                input_array_with_history = np.concatenate([input_array_state, input_array_history], axis=0)
                
                input_array = np.rollaxis(input_array_with_history,  1, 0)
                input_array = input_array.reshape((54, self.history * 6))
                input_list.append(input_array)
                
                action = np.argmax(policy)
                next_cube = cube.copy()
                next_cube.step([action])
                
            inputs = np.array(input_list)

        return inputs, policies, values


class ConvModel(BaseModel): 
    """
    A residual 2D-convolutional model.
    """   

    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False, history=1):
        BaseModel.__init__(self, use_cache, max_cache_size, rotationally_randomize, history)
        assert history == 1, "history > 1 not yet implemented for ConvModel"

        self.input_shape = (6 * 6, 3, 3)

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

        self._build(model)

    def process_single_input(self, input_array):
        input_array = input_array.reshape((-1, 54, 6))
        input_array = np.rollaxis(input_array, 2, 1).reshape(-1, 6*6, 3, 3)
        return input_array

    def process_training_data(self, inputs, policies, values, augment=True):
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


class ConvModel2D3D(BaseModel): 
    """
    A residual 3D convolutional model restricted to the 2D boundary of the cube.
    """   

    def __init__(self, use_cache=True, max_cache_size=10000, rotationally_randomize=False, history=1):
        BaseModel.__init__(self, use_cache, max_cache_size, rotationally_randomize, history)
        self.input_shape = (54, self.history * 6)

    def build(self):
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

        from model_constant_arrays import x3d, y3d, z3d, neighbors
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
            assert in_tensor.shape[1] == 54, in_tensor.shape

            # pad (output dim: None x 55 x ?)
            padded = Lambda(lambda x: K.temporal_padding(x, (0, 1)))(in_tensor) # just pad end
            assert padded.shape[1] == 55, padded.shape
            
            # align neighbors (output dim: None x 54 x 27 x ?)
            #aligned = K.gather(padded, neighbors)
            #aligned = padded[ neighbors[np.newaxis].astype(np.int32), :]
            aligned = Lambda(lambda x: tf.gather(x, neighbors, axis=1))(padded)
            assert aligned.shape[1:3] == (54, 27), aligned.shape
            
            # 2D convolution in one axis (output dim: None x 54 x 1 x filter_size)
            conv = Conv2D(filter_size, kernel_size=(1, 27), 
                          strides=(1, 1), 
                          padding='valid', 
                          data_format="channels_last",
                          kernel_regularizer=l2(0.001), 
                          bias_regularizer=l2(0.001))(aligned)
            assert conv.shape[1:3] == (54, 1), conv.shape

            # reshape (output dim: None x 54 x filter_size)
            out_tensor = Lambda(lambda x: K.squeeze(x, axis=2))(conv)
            assert out_tensor.shape[1] == 54, out_tensor.shape

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
        state_input = Input(shape=self.input_shape, name='state_input')
        
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

        self._build(model)

    def process_single_input(self, input_array):
        input_array = input_array.reshape((self.history, 54, 6))
        if self.history > 1:
            input_array = np.rollaxis(input_array,  1, 0)
            input_array = input_array.reshape((1, 54, self.history * 6))
        return input_array

    def process_training_data(self, inputs, policies, values, augment=True):
        """
        Convert training data to arrays.  
        Augment data
        Reshape to fit model input.
        """
        
        # augment with all 48 color rotations
        if augment:
            inputs, policies, values = augment_data(inputs, policies, values)

        # process arrays now to save time during training
        if self.history == 1:
            inputs = inputs.reshape((-1, 54, 6))
        else:
            # use that the inputs are in order to attach the history
            # use the policy/input match to determine when we reached a new game
            next_cube = None
            input_array_with_history = None
            input_list = []
            for state, policy in zip(inputs, policies):
                cube = BatchCube()
                cube.load_bit_array(state)
                
                if next_cube is None or cube != next_cube:
                    # blank history
                    input_array_history = np.zeros((self.history-1, 54, 6), dtype=bool)
                else:
                    input_array_history = input_array_with_history[:-1]
                
                input_array_state = state.reshape((1, 54, 6))
                input_array_with_history = np.concatenate([input_array_state, input_array_history], axis=0)
                
                input_array = np.rollaxis(input_array_with_history,  1, 0)
                input_array = input_array.reshape((54, self.history * 6))
                input_list.append(input_array)
                
                action = np.argmax(policy)
                next_cube = cube.copy()
                next_cube.step([action])
                
            inputs = np.array(input_list)

        return inputs, policies, values

