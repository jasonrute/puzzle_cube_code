"""
An implementation of an array Rubic's cubes.

Actions are stored in this order (corresponding to these numbers)
    0:L, 1:L', 2:R, 3:R', 4:U, 5:U', 6:D, 7:D', 8:F, 9:F', 10:B, 11:B'

Internally the cube is stored as an integer array of dimension ? x 54
    One row for each cube, stored in a way easy to import into PyCuber (for printing)
    The numbers 0-5 correspond to the colors RYGWOB
    The nth block of 9 numbers corresponds to the nth face (in the order RYGWOB)
    E.g. the solved cube is stored as 
        000000000011111111111222222222233333333334444444444555555555

When it is outputed to a bit array (for the NN) it is stored as ? x 54 x 6
"""

import numpy as np
import pycuber as pc

basic_moves = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

# store cubes via 
#    (-1, 6 faces, 9 squares) arrays
#    (-1, 6 colors, 6 faces, 9 squares) bit array

solved_cube_list = np.array([0]*9 + [1]*9 + [2]*9 + [3]*9 + [4]*9 + [5]*9)
solved_cube_bit_array = np.eye(6)[np.newaxis, solved_cube_list]

pc_indices = [58, 61, 64, 95, 98, 101, 132, 135, 138, 10, 13, 16, 29, 32, 35, 48, 51, 54, 67, 70, 73, 104, 107, 110, 141, 144, 147, 178, 181, 184, 197, 200, 203, 216, 219, 222, 76, 79, 82, 113, 116, 119, 150, 153, 156, 85, 88, 91, 122, 125, 128, 159, 162, 165]

colors = ['r', 'y', 'g', 'w', 'o', 'b']
color_dict = {'r': 0, 'y': 1, 'g': 2, 'w': 3, 'o': 4, 'b': 5}

# Here forward_action_array[a, n] is the position where square n moves to under action number a
forward_action_array =\
    np.array([[ 2,  5,  8,  1,  4,  7,  0,  3,  6, 18, 10, 11, 21, 13, 14, 24, 16,
        17, 27, 19, 20, 30, 22, 23, 33, 25, 26, 53, 28, 29, 50, 31, 32, 47,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 15, 48, 49, 12,
        51, 52,  9],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 51, 12, 13, 48, 15, 16,
        45, 18, 19, 11, 21, 22, 14, 24, 25, 17, 27, 28, 20, 30, 31, 23, 33,
        34, 26, 38, 41, 44, 37, 40, 43, 36, 39, 42, 35, 46, 47, 32, 49, 50,
        29, 52, 53],
       [45, 46, 47,  3,  4,  5,  6,  7,  8, 11, 14, 17, 10, 13, 16,  9, 12,
        15,  0,  1,  2, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 18, 19, 20, 39, 40, 41, 42, 43, 44, 36, 37, 38, 48, 49, 50,
        51, 52, 53],
       [ 0,  1,  2,  3,  4,  5, 24, 25, 26,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 42, 43, 44, 29, 32, 35, 28, 31, 34, 27,
        30, 33, 36, 37, 38, 39, 40, 41, 51, 52, 53, 45, 46, 47, 48, 49, 50,
         6,  7,  8],
       [ 0,  1, 17,  3,  4, 16,  6,  7, 15,  9, 10, 11, 12, 13, 14, 36, 39,
        42, 20, 23, 26, 19, 22, 25, 18, 21, 24,  2,  5,  8, 30, 31, 32, 33,
        34, 35, 29, 37, 38, 28, 40, 41, 27, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53],
       [33,  1,  2, 34,  4,  5, 35,  7,  8,  6,  3,  0, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 44,
        41, 38, 36, 37,  9, 39, 40, 10, 42, 43, 11, 47, 50, 53, 46, 49, 52,
        45, 48, 51],
       [ 6,  3,  0,  7,  4,  1,  8,  5,  2, 53, 10, 11, 50, 13, 14, 47, 16,
        17,  9, 19, 20, 12, 22, 23, 15, 25, 26, 18, 28, 29, 21, 31, 32, 24,
        34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 33, 48, 49, 30,
        51, 52, 27],
       [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 20, 12, 13, 23, 15, 16,
        26, 18, 19, 29, 21, 22, 32, 24, 25, 35, 27, 28, 51, 30, 31, 48, 33,
        34, 45, 42, 39, 36, 43, 40, 37, 44, 41, 38, 17, 46, 47, 14, 49, 50,
        11, 52, 53],
       [18, 19, 20,  3,  4,  5,  6,  7,  8, 15, 12,  9, 16, 13, 10, 17, 14,
        11, 36, 37, 38, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35, 45, 46, 47, 39, 40, 41, 42, 43, 44,  0,  1,  2, 48, 49, 50,
        51, 52, 53],
       [ 0,  1,  2,  3,  4,  5, 51, 52, 53,  9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23,  6,  7,  8, 33, 30, 27, 34, 31, 28, 35,
        32, 29, 36, 37, 38, 39, 40, 41, 24, 25, 26, 45, 46, 47, 48, 49, 50,
        42, 43, 44],
       [ 0,  1, 27,  3,  4, 28,  6,  7, 29,  9, 10, 11, 12, 13, 14,  8,  5,
         2, 24, 21, 18, 25, 22, 19, 26, 23, 20, 42, 39, 36, 30, 31, 32, 33,
        34, 35, 15, 37, 38, 16, 40, 41, 17, 43, 44, 45, 46, 47, 48, 49, 50,
        51, 52, 53],
       [11,  1,  2, 10,  4,  5,  9,  7,  8, 38, 41, 44, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,  0,
         3,  6, 36, 37, 35, 39, 40, 34, 42, 43, 33, 51, 48, 45, 52, 49, 46,
        53, 50, 47]])

# Unfortionately the forward_action_array is in the wrong order.  
# It is in the order L' R' U' D' F' B' L R U D F B
# But we want it in the order L L' R' U' D' F' B' L R U D F B
# Here we fix it.

action_array = forward_action_array[[6, 0, 7, 1, 8, 2, 9, 3, 10, 4, 11, 5]]

class BatchCube():
    """
    An implementation of a vector of Rubik's cubes using NumPy arrays.
    It has methods for converting to the PyCuber object class.
    """
    
    def __init__(self, length = 1):
        """
        Creates length-many solved cubes
        """
        self._cube_array = np.repeat(solved_cube_list[np.newaxis], repeats=length, axis=0)
    
    def copy(self):
        new_cubes = BatchCube(len(self))
        new_cubes._cube_array = self._cube_array.copy()
        return new_cubes
        
    def __len__(self):
        return self._cube_array.shape[0]
     
    def bit_array(self):
        return np.eye(6, dtype=bool)[self._cube_array]

    def step(self, actions):
        """
        Assuming actions is a list of length = len(self)
        """
        action_indices = action_array[actions]
        sample_index, _ = np.indices(self._cube_array.shape)
        self._cube_array = self._cube_array[sample_index, action_indices]

    def step_independent(self, actions):
        """
        Performs all actions independently on each state
        """
        action_len = len(actions)
        cubes_len = len(self._cube_array)
        
        self._cube_array = np.repeat(self._cube_array, repeats=action_len, axis=0)
        actions = np.tile(actions, cubes_len)
        
        self.step(actions)

    def randomize(self, dist=100):
        l = len(self._cube_array)
        for _ in range(dist):
            actions = np.random.choice(12, l)
            self.step(actions)

    def to_pycuber(self):
        """
        Returns a list of PyCuber Cubes
        """
        # convert to number representation
        return [pc.Cube(pc.helpers.array_to_cubies([str(int(c)) for c in color_list])) 
                for color_list in self._cube_array]
        
    def from_pycuber(self, pc_list):
        """
        Takes in a list of PyCuber Cubes and converts them to an array.
        It overwrites the array in the object.
        """
        
        # The only good way I know to do this is to first convert the pc.Cube object
        # to a string and read off the values.
        
        pre_array = []
        for cube in pc_list:
            s = str(c)
            cube_array = np.array([color_dict[s[i]] for i in pc_indices])
            pre_array.append(cube_array)
        self._cube_array = np.array(pre_array)
    
    def done(self):
        return (self._cube_array == solved_cube_list).all(axis=1)
    
    def remove_done(self):
        self._cube_array = self._cube_array[~self.done()]
    
    def __str__(self):
        return "".join(str(c) for c in self.to_pycuber())

if __name__ == '__main__':
    # blank_cube and export
    bc = BatchCube()
    for c in bc.to_pycuber():
        print(c)
    
    
    # import blank_cube and export it
    bc = BatchCube()
    bc.from_pycuber([pc.Cube()])
    print(len(bc))
    assert len(bc) == 1
    for c in bc.to_pycuber():
        print(c)
        
    
    # test all actions
    for i, m in enumerate(basic_moves):
        bc = BatchCube()
        mycube = pc.Cube()
        
        bc.step(np.array([i]))
        mycube.perform_step(m)
        
        for c in bc.to_pycuber():
            print(i, m, c == mycube)
            assert c == mycube, "Action {} (in BatchCube) does not match action {} in PyCuber.  The first gives:\n{}\nThe second gives:\n{}".format(i, m, c, mycube)
    
    # multiple actions
    bc = BatchCube(12)
    cube_list = [pc.Cube() for _ in range(12)]
    
    bc.step(np.arange(12)) # perform all twelve actions in parallel
    for c, m in zip(cube_list, basic_moves):
        c.perform_step(m)
    
    for c1, c2 in zip(bc.to_pycuber(), cube_list):
        print(c1 == c2)
        assert c1 == c2
    
    # test remove_done
    bc = BatchCube(12)
    
    bc.step(np.arange(12)) # perform all twelve actions in parallel
    bc.step(np.array([0]*12))
    
    print(bc.done())
    assert bc.done().sum() == 1
    assert bc.done()[1] == 1
    bc.remove_done()
    print(len(bc))
    assert len(bc) == 11

    # test independent actions
    bc.step_independent(np.arange(12))
    bc.step_independent(np.arange(12))
    print(len(bc))
    assert len(bc) == 11 * 12 * 12
    print(bc.done().sum())
    assert bc.done().sum() == 14

    #test randomize
    bc = BatchCube(2)
    bc.randomize(1)
    print("Randomized once:")
    for c in bc.to_pycuber():
        print(c)
    print("Randomized many times:")
    bc.randomize()
    for c in bc.to_pycuber():
        print(c)
    


    