import numpy as np

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

def starting_model3d():
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
        
        # pad (output dim: None x 55 x ?)
        padded = Lambda(lambda x: K.temporal_padding(x, (0, 1)))(in_tensor) # just pad end
        
        # align neighbors (output dim: None x 54 x 27 x ?)
        #aligned = K.gather(padded, neighbors)
        #aligned = padded[ neighbors[np.newaxis].astype(np.int32), :]
        aligned = Lambda(lambda x: tf.gather(x, neighbors, axis=1))(padded)
        
        # 2D convolution in one axis (output dim: None x 54 x 1 x filter_size)
        conv = Conv2D(filter_size, kernel_size=(1, 27), 
                      strides=(1, 1), 
                      padding='valid', 
                      data_format="channels_last",
                      kernel_regularizer=l2(0.001), 
                      bias_regularizer=l2(0.001))(aligned)

        
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
                  optimizer=Adam(lr=.001))

    return model

def starting_model2d():
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
                  optimizer=Adam(lr=.001))

    return model

import threading
import queue 

class Task:
    def __init__(self):
        self.lock = threading.Condition()
        self.input = None
        self.output = None

class BatchProcessHelper:
    def __init__(self):
        self.lock = threading.RLock()
        self._batch_size = 5
    def get_batch_size(self):
        with self.lock:
            return self._batch_size
    def decrement_batch_size(self):
        with self.lock:
            self._batch_size -= 1
    def set_batch_size(self, batch_size):
        with self.lock:
            self._batch_size = batch_size

input_queue = queue.Queue()

def get_value(input_value):
    task = Task()

    # put the value on the queue to be processed
    task.input = input_value
    with task.lock:
        input_queue.put(task) # put task on queue to be processed
        task.lock.wait() # wait until task is processed
        return task.output # return output

def batch_process(get_output, batch_process_helper):
    import numpy as np
    task_list = []

    while True:
        # retrieve items from the queue
        task = input_queue.get()
        task_list.append(task)

        if len(task_list) >= batch_process_helper.get_batch_size():
            array = np.array([task.input.squeeze(axis=0) for task in task_list])
            policies, values = get_output([array, 0])

            for p, v, task in zip(policies, values, task_list):
                with task.lock:
                    task.output = [p, v]
                    task.lock.notify() # mark as being complete

            task_list = []

if __name__ == '__main__':
    import time
    from keras import backend as K

    model = starting_model3d()
    get_output = K.function([model.input, K.learning_phase()], [model.output[0], model.output[1]])

    batch_process_helper = BatchProcessHelper()
    worker = threading.Thread(target=batch_process, args=(get_output, batch_process_helper,))
    worker.daemon = True
    worker.start()

    # just use random data
    inputs = np.random.choice(2, size=(2**10, 54, 6), p=[48/54, 6/54]).astype(bool)

    for i in range(11):
        batch_size = 2**i
        print()
        print("batch size:", 2**i)

        my_inputs = inputs.copy().reshape((-1, batch_size, 54, 6))
        
        t1 = time.time()
        for batch in my_inputs:
            get_output([batch, 0])
        print("get_output          ", "time:", time.time() - t1)
        t1 = time.time()
        for batch in my_inputs:
            model.predict(batch)
        print("predict             ", "time:", time.time() - t1)
        t1 = time.time()
        for batch in my_inputs:
            model.predict(batch, batch_size=2**i)
        print("predict(batch_size=)", "time:", time.time() - t1)

        my_inputs = inputs.copy().reshape((-1, 1, 54, 6))
        batch_process_helper.set_batch_size(batch_size)

        t1 = time.time()
        threads = []
        for batch in my_inputs:
            t = threading.Thread(target=get_value, args=(batch, ))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print("get_value           ", "time:", time.time() - t1)
