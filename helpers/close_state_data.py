"""
This script generates and saves the states closest to the solved state.  

It stores the results as an hd5 file as an array.  See the code at the end
for how to load the data.

It stores three values (stored in parallel arrays).

- A bit array representing the state.

- A boolean array which marks all actions which lead to the shortest path to the solved cube

- The distance to the solved_cube
"""

import collections
import numpy as np

# Load BatchCube
import sys
sys.path.append('..') # add parent directory to path
from batch_cube import BatchCube


MAX_DISTANCE = 6

eye12 = np.eye(12, dtype=bool)

state_dict = {} # value =  (best_actions, distance)


print("Generating data...")
# start with solved cube
cubes = BatchCube(1)
solved_cube = cubes._cube_array[0]
key = cubes.bit_array().tobytes()
best_actions = np.zeros(12)
distance = 0
state_dict[key] = (cubes.bit_array()[0], best_actions, distance)

size = 1
for distance in range(1, MAX_DISTANCE+1):
    print("Distance:", distance)
    
    # go to neighbors
    cubes.step_independent(np.arange(12))

    # record last move taken
    last_action = np.tile(np.arange(12), size)

    # find inverse of that move (using ^)
    best_action = last_action ^ 1
    best_action_one_hot = eye12[best_action]

    # record data
    internal_array = cubes._cube_array
    bit_array = cubes.bit_array()

    temp_dict = {}
    new_cube_array = []
    for bits, internal, action in zip(bit_array, internal_array, best_action_one_hot):
        key = bits.tobytes()
        
        if key in state_dict:
            continue

        if key in temp_dict:
            best_actions = temp_dict[key][1]
            best_actions += action
            continue

        temp_dict[key] = (bits, action, distance)
        new_cube_array.append(internal)

    state_dict.update(temp_dict)
    size = len(state_dict)
    
    print("total:", size, "current:", len(new_cube_array))

    # rebuild cube
    cubes._cube_array = np.array(new_cube_array)


print("Storing data...")

# convert to arrays for easy storing
keys = []
bits = []
best_actions = []
distances = []

for b, a, d in state_dict.values():
    bits.append(b[np.newaxis])
    best_actions.append(a)
    distances.append(d)

print(bits[0].shape)
bits = np.concatenate(bits, axis=0)
best_actions = np.array(best_actions, dtype=bool)
distances = np.array(distances)

# put into hD5 file
import h5py
h5f = h5py.File('../save/close_state_data.h5', 'w')
h5f.create_dataset('bits', data=bits)
h5f.create_dataset('best_actions', data=best_actions)
h5f.create_dataset('distances', data=distances)
h5f.close()

     
print("Load data...")

# Load
h5f = h5py.File('../save/close_state_data.h5', 'r')
bits = h5f['bits'][:]
best_actions = h5f['best_actions'][:]
distances = h5f['distances'][:]
h5f.close()

# Rebuild dictionary
state_dict = {b.tobytes():(b, a, int(d)) for b, a, d in zip(bits, best_actions, distances)}


print("Testing data...")
# Test data types
for k, v in state_dict.items():
    assert v[0].dtype == bool
    assert v[1].dtype == bool
    break

# Test data
import numpy as np
for i in range(1000):
    test_cube = BatchCube(1)
    test_cube.randomize(1 + (i % MAX_DISTANCE))
    _, best_actions, distance = state_dict[test_cube.bit_array().tobytes()]

    for _ in range(distance):
        assert not test_cube.done()[0]
        action = np.random.choice(12, p=best_actions/np.sum(best_actions))
        test_cube.step([action])
        _, best_actions, _ = state_dict[test_cube.bit_array().tobytes()]

    assert test_cube.done()[0]

print("Passed all tests")
        
        





