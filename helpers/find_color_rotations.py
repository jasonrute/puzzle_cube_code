"""
Find all 6*4*2 = 48 color rotations

6 ways to rotation red (to any other color A in RYGWOB)
4 ways to rotate yellow (to any adjacent color B adjacent to A:)
2 ways to flip green (to any color C adjacent to A and B:)
the rest follow (orange is mapped opposite A, white is mapped opposite B, and blue is mapped opposite green)
"""

import numpy as np 
from pprint import pprint

# Load BatchCube
import sys
sys.path.append('..') # add parent directory to path
from batch_cube import BatchCube


# colors are stored in the (bad) order: 
# RYGWOB
# 012345
# 
#          [y][y][y]
#          [y][y][y]
#          [y][y][y]
# [r][r][r][g][g][g][o][o][o][b][b][b]
# [r][r][r][g][g][g][o][o][o][b][b][b]
# [r][r][r][g][g][g][o][o][o][b][b][b]
#          [w][w][w]
#          [w][w][w]
#          [w][w][w]
# 
opposite_color = {0:4, 1:3, 2:5, 3:1, 4:0, 5:2}
first_cw_corner = (0, 1, 2)

adjacent_colors = {c1: {c for c in range(6) if c != c1 and c != opposite_color[c1]} for c1 in range(6)}
#pprint(adjacent_colors)
adjacent_to_color_pair = {(c1, c2): adjacent_colors[c1] & adjacent_colors[c2] for c1 in range(6) for c2 in range(6) if c1 in adjacent_colors[c2] and (adjacent_colors[c1] & adjacent_colors[c2])}
#pprint(adjacent_to_color_pair)
"""
adjacent_color_cw = {(0,1):2, (0,2):3, (0,3):5, (0,5):1, (1,0):5, (2,0):1, (3,0):2, (5,0):3,
                     (1,2):0, (2,1):4, (1,4):2, (4,1):5, (1,5):4, (5,1):0,
                     (2,3):0, (3,2):4, (2,4):3, (4,2):1,
                     (3,4):5, (4,3):2, (3,5):0, (5,3):4,
                     (4,5):3, (5,4):1}
"""
# construct automaticially encase I change the color encoding later
adjacent_color_cw = {first_cw_corner[:2]: first_cw_corner[2]} # seed with one cw corner
while len(adjacent_color_cw) < 24: # 24 = 6 sides * 4 adjacent sides
    for a,b in list(adjacent_color_cw):
        c = adjacent_color_cw[a,b]
        adjacent_color_cw[b,c]=a
        adjacent_color_cw[c,a]=b
        adjacent_color_cw[b,a]=opposite_color[c]
#pprint(adjacent_color_cw)

# check that it is correct
assert adjacent_color_cw[0,1]==2
for a,b in adjacent_color_cw:
    assert adjacent_to_color_pair[a,b] == {adjacent_color_cw[a,b], adjacent_color_cw[b,a]}, \
           (a, b, adjacent_to_color_pair[a,b], {adjacent_color_cw[a,b], adjacent_color_cw[b,a]})

    c = adjacent_color_cw[a,b]
    assert adjacent_color_cw[b,c]==a
    assert adjacent_color_cw[c,a]==b
    assert adjacent_color_cw[b,a]==opposite_color[c]

# actions are stored in the order:
# 0:L, 1:L', 2:R, 3:R', 4:U, 5:U', 6:D, 7:D', 8:F, 9:F', 10:B, 11:B'
# and are associated with colors as in a confusing manner
"""
"L": "red", "0": "red", "red": "red", 
"U": "yellow", "1": "yellow", "yellow": "yellow", 
"F": "green", "2": "green", "green": "green", 
"D": "white", "3": "white", "white": "white", 
"R": "orange", "4": "orange", "orange": "orange", 
"B": "blue", "5": "blue", "blue": "blue",
"""
action_from_color = [0, 4, 8, 6, 2, 10]
opp_actions = [(i^1) for i in range(12)]

action_from_color_rotation = {(c,r): action_from_color[c] if r else opp_actions[action_from_color[c]] 
                              for c in range(6) for r in [False, True]}
color_rotation_from_action = {a: pair for pair, a in action_from_color_rotation.items()}
color_from_action = [color_rotation_from_action[a][0] for a in range(12)]
rotation_from_action = [color_rotation_from_action[a][1] for a in range(12)]

"""
Every square position on the cube can be descriped by its starting color and the set of actions 
which preserve it (or equivalently those which move it).  This is also the same as decribing
a position by its starting color and the adjacent starting colors.
"""
color_encoding = []
bc = BatchCube(1)
starting_colors = list(bc._cube_array[0])


# replace colors with positions
bc._cube_array[0] = np.arange(54)

neighbor_colors = [set() for _ in range(54)]
for c in range(6):
    a = action_from_color[c]
    bc.step(a)
    c_adjacent = [i for i in range(54) if bc._cube_array[0][i] != i and starting_colors[i] != c] 
    for i in range(54):
        if i in c_adjacent:
            neighbor_colors[i].add(c)

    bc.step(opp_actions[a]) #undo action

color_encoding = [(c, frozenset(s)) for c, s in zip(starting_colors, neighbor_colors)]
color_decoding = {(c, frozenset(s)):i for i, c, s in zip(range(54), starting_colors, neighbor_colors)}

assert len(color_encoding) == 54
assert len(color_decoding) == 54
for i in range(54):
    assert color_decoding[color_encoding[i]] == i

#pprint(color_encoding)
"""
We can also think of this in terms of actions.  The red-face/L-action can be mapped
to any of the 6 faces/6 basic action pairs (L/L' R/R' U/U' D/D' F/F' B/B').  The yellow-face/U-action 
can be mapped to any of the 4 adjacent faces/actions.  Finally one can choose the remaining direction.
If the direction is positive, then non inverse actions map to non-invers actions and
green maps to the positive third color.  (Else, the negative)
"""
color_permutations = []
opp_color_permutations = []
action_permutations = []
opp_action_permutations = []
position_permutations = []
opp_position_permutations = []
for new_r in range(6): # decide where red goes
    for new_y in adjacent_colors[new_r]: # decide where yellow goes
        for reflect in [False, True]: # decide if reflecting (cw <-> ccw)
            if not reflect:
                new_g = adjacent_color_cw[new_r, new_y]
            else:
                new_g = adjacent_color_cw[new_y, new_r]
            new_o = opposite_color[new_r]
            new_w = opposite_color[new_y]
            new_b = opposite_color[new_g]

            color_perm = np.array([new_r, new_y, new_g, new_w, new_o, new_b])
            opp_color_perm = np.arange(6)
            opp_color_perm[color_perm] = np.arange(6)

            action_perm = np.array([action_from_color_rotation[color_perm[c], r ^ reflect] 
                                    for c, r in zip(color_from_action, rotation_from_action)])
            opp_action_perm = np.arange(12)
            opp_action_perm[action_perm] = np.arange(12)

            position_perm = np.array([color_decoding[color_perm[c], frozenset(color_perm[cs] for cs in s)]
                             for (c, s) in color_encoding])
            opp_position_perm = np.arange(54)
            opp_position_perm[position_perm] = np.arange(54)

            color_permutations.append(color_perm)
            opp_color_permutations.append(opp_color_perm)
            action_permutations.append(action_perm)
            opp_action_permutations.append(opp_action_perm)
            position_permutations.append(position_perm)
            opp_position_permutations.append(opp_position_perm)

color_permutations = np.array(color_permutations)
opp_color_permutations = np.array(opp_color_permutations)
action_permutations = np.array(action_permutations)
opp_action_permutations = np.array(opp_action_permutations)
position_permutations = np.array(position_permutations)
opp_position_permutations = np.array(opp_position_permutations)

def swap_colors_1(bc, i):
    idx = np.indices(bc._cube_array.shape)[0]
    bc._cube_array = bc._cube_array[idx, position_permutations[i][np.newaxis]]
    bc._cube_array = opp_color_permutations[i][bc._cube_array]

def swap_colors_2(bc, i):
    bit_array = bc.bit_array()
    idx = np.indices(bc.bit_array().shape)[0]
    pos_perm = position_permutations[i][np.newaxis,:,np.newaxis]
    col_perm = color_permutations[i][np.newaxis, np.newaxis]
    bit_array = bit_array[idx, pos_perm, col_perm]
    bc.load_bit_array(bit_array)

if __name__ == '__main__':
    print("color_permutations = \\")
    pprint(np.array(color_permutations))
    print()

    print("position_permutations = \\")
    np.set_printoptions(threshold=np.inf) # allows one to see the whole array even if it is big
    pprint(np.array(position_permutations))
    print()

    print("opp_action_permutations = \\")
    pprint(np.array(opp_action_permutations))
    print()

    print("action_permutations = \\")
    pprint(np.array(action_permutations))

    # test opposites
    for i in range(48):
        bc0 = BatchCube()
        bc0.randomize(100)
        bc = bc0.copy()
        
        bc._cube_array[0] = bc._cube_array[0][position_permutations[i]]
        bc._cube_array[0] = bc._cube_array[0][opp_position_permutations[i]]
        assert bc == bc0

        bc._cube_array[0] = color_permutations[i][bc._cube_array[0]]
        bc._cube_array[0] = opp_color_permutations[i][bc._cube_array[0]]
        assert bc == bc0

        policy0 = np.random.uniform(size=12)
        policy1 = policy0.copy()
        policy1 = policy1[action_permutations[i]]
        policy1 = policy1[opp_action_permutations[i]]
        assert np.array_equal(policy0, policy1)

    # test solved cube under permuations
    for i in range(48):
        bc0 = BatchCube()
        bc = bc0.copy()
        
        bc._cube_array[0] = bc._cube_array[0][position_permutations[i]]
        bc._cube_array[0] = opp_color_permutations[i][bc._cube_array[0]]
        assert bc == bc0

    # test solved cube under permuations (bit array)
    for i in range(48):
        bc0 = BatchCube()
        bc = bc0.copy()
        
        bit_array = bc.bit_array()
        idx = np.indices(bc.bit_array().shape)[0]
        pos_perm = position_permutations[i][np.newaxis,:,np.newaxis]
        col_perm = color_permutations[i][np.newaxis, np.newaxis]
        bit_array = bit_array[idx, pos_perm, col_perm]
        bc.load_bit_array(bit_array)
        assert bc == bc0

    # test swap functions
    for i in range(48):
        bc1 = BatchCube(10)
        bc1.randomize(100)
        bc2 = bc1.copy()

        swap_colors_1(bc1, i)
        swap_colors_2(bc2, i)

        assert bc1 == bc2, "\n" + str(bc1) + "\n" + str(bc2)

    """
    # test action permuations
    for i in range(48):
        for a in range(12):
            bc1 = BatchCube(1)
            bc1.randomize(100)
            bc2 = bc1.copy()
            
            # b1: switch colors and then preform switched action
            swap_colors_2(bc1, i)
            new_a = opp_action_permutations[i][a]
            bc1.step(new_a)

            # preform action and then swap colors
            bc2.step(a)
            swap_colors_2(bc2, i)

            # then do new action
            assert bc1 == bc2, "\n" + str(bc1) + "\n" + str(bc2)
    """
    # test action permuations
    for i in range(48):
        bc1 = BatchCube(12)
        bc1.randomize(100)
        bc2 = bc1.copy()
        
        # b1: switch colors and then preform switched action
        swap_colors_2(bc1, i)
        bc1.step(opp_action_permutations[i])

        # preform action and then swap colors
        bc2.step(np.arange(12))
        swap_colors_2(bc2, i)

        # then do new action
        assert bc1 == bc2, "\n" + str(bc1) + "\n" + str(bc2)

    # test opposite action permuations
    for i in range(48):
        bc1 = BatchCube(12)
        bc1.randomize(100)
        bc2 = bc1.copy()
        
        # b1: switch colors and then preform switched action
        swap_colors_2(bc1, i)
        bc1.step(np.arange(12))

        # preform action and then swap colors
        bc2.step(action_permutations[i])
        swap_colors_2(bc2, i)

        # then do new action
        assert bc1 == bc2, "\n" + str(bc1) + "\n" + str(bc2)

    # test rearranging policy
    for i in range(48):
        for _ in range(10):
            bc1 = BatchCube(1)
            bc1.randomize(100)
            bc2 = bc1.copy()
            
            # b1: find policy on swapped side
            swap_colors_2(bc1, i)
            policy = np.random.uniform(size=12)
            bc1.step(np.argmax(policy))

            # b2: convert policy back to normal side
            bc2.step(np.argmax(policy[opp_action_permutations[i]]))
            swap_colors_2(bc2, i)

            # then do new action
            assert bc1 == bc2


    # test augmenting data set
    size = 100
    bc1 = BatchCube(size)
    bc1.randomize(100)
    bc2 = bc1.copy()

    # find policy and value on original rotation
    inputs = bc1.bit_array()
    policies = original_policies = np.random.uniform(size=(size, 12))
    values = original_values = np.random.uniform(size=size)
    
    bc1.step(np.argmax(policies, axis=1))

    # convert state, policies, and values to all rotations
    inputs = np.array(inputs).reshape((-1, 54, 6))
    sample_size = inputs.shape[0]

    sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis, np.newaxis]
    pos_perm = position_permutations[:, np.newaxis, :, np.newaxis]
    col_perm = color_permutations[:, np.newaxis, np.newaxis, :]
    inputs = inputs[sample_idx, pos_perm, col_perm]
    inputs = inputs.reshape((-1, 54, 6))

    policies = np.array(policies).reshape((-1, 12))
    sample_idx = np.arange(sample_size)[np.newaxis, :, np.newaxis]
    action_perm = action_permutations[:, np.newaxis, :]
    policies = policies[sample_idx, action_perm]
    policies = policies.reshape((-1, 12))

    values = np.array(values).reshape((-1, ))
    values = np.tile(values, 48)

    # test that this is the right permuation
    assert len(inputs) == len(policies)
    assert len(inputs) == len(values)

    inputs = inputs.reshape(48, -1, 54, 6)
    policies = policies.reshape(48, -1, 12)
    values = values.reshape(48, -1)
    
    for rot in range(48):
        i2 = inputs[rot]
        p2 = policies[rot]
        v2 = values[rot]
        
        assert v2.shape == original_values.shape
        assert np.array_equal(v2, original_values)
        
        sample_idx = np.arange(len(p2))[:, np.newaxis]
        p1 = p2[sample_idx, opp_action_permutations[rot][np.newaxis]]
        assert p1.shape == original_policies.shape, str(rot) + str(p1.shape) + " " + str(original_policies.shape)
        assert np.array_equal(p1, original_policies), str(rot) + "\n" + str(p1) + "\n" + str(original_policies)
        
        bc2_ = BatchCube()
        bc2_.load_bit_array(i2)
        bc2_.step(np.argmax(p2, axis=1))

        bc1_ = bc1.copy()
        swap_colors_2(bc1_, rot)
        assert bc1_ == bc2_, "\n" + str(bc1_) + "\n" + str(bc2_)







