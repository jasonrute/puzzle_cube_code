"""
Figure out how to embedd 54 positions of the Cube 

          [y][y][y]
          [y][y][y]
          [y][y][y]
 [r][r][r][g][g][g][o][o][o][b][b][b]
 [r][r][r][g][g][g][o][o][o][b][b][b]
 [r][r][r][g][g][g][o][o][o][b][b][b]
          [w][w][w]
          [w][w][w]
          [w][w][w]

into 3d using the following encoding (- means blank)
 -  -  -  -  -  |  - [b][b][b] -  |   - [b][b][b] -  |   - [b][b][b] -  |  -  -  -  -  -
 - [y][y][y] -  | [r] -  -  - [o] |  [r] -  -  - [o] |  [r] -  -  - [o] |  - [w][w][w] -
 - [y][y][y] -  | [r] -  -  - [o] |  [r] -  -  - [o] |  [r] -  -  - [o] |  - [w][w][w] -
 - [y][y][y] -  | [r] -  -  - [o] |  [r] -  -  - [o] |  [r] -  -  - [o] |  - [w][w][w] -
 -  -  -  -  -  |  - [g][g][g] -  |   - [g][g][g] -  |   - [g][g][g] -  |  -  -  -  -  - 

Simple rules:
  base colors:
    r/o -> x = 0/4
    y/w -> y = 0/4
    g/b -> z = 0/4
  adjacencies:
    r/o -> x = 1/3
    y/w -> y = 1/3
    g/b -> z = 1/3
  no adjacency:
    r/o -> x = 2/2
    y/w -> y = 2/2
    g/b -> z = 2/2
"""

import numpy as np 
from pprint import pprint

from find_color_rotations import opposite_color, first_cw_corner, color_encoding, color_decoding, swap_colors_1

# Load BatchCube
import sys
sys.path.append('..') # add parent directory to path
from batch_cube import BatchCube

# orientation choices:
# opposite_color = {0:4, 1:3, 2:5, 3:1, 4:0, 5:2}
# first_cw_corner = (0, 1, 2) #x, y, z axis respectively

axis_from_color = {c: i for i, c in enumerate(first_cw_corner)}
axis_from_color.update({opposite_color[c]: i for i, c in enumerate(first_cw_corner)})

x = []
y = []
z = []
coords = (x, y, z)

for i in range(54):
    color, adj_colors = color_encoding[i]
    
    for c in range(6):
        axis = axis_from_color[c]
        sign = 1 if c in first_cw_corner else -1

        if c == color:
            dist = 2
        elif c in adj_colors:
            dist = 1
        elif (c in first_cw_corner) and \
             (opposite_color[c] not in adj_colors) and \
             (opposite_color[c] != color):
            dist = 0
        else:
            continue
        
        coords[axis].append(2 - sign * dist)

x,y,z = np.array(x), np.array(y), np.array(z) 

# Find neighbors by direction:
# - number of directions is 27 = 3x3x3

square_indices = np.full((6, 6, 6), -1, dtype=int) # -1 means not a square, dimension 6 = 5 + 1 for wrapping
square_indices[x, y, z] = np.arange(54)

neighbors = []
for pos in range(54):
    x0 = x[pos]
    y0 = y[pos]
    z0 = z[pos]

    my_neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dz in [-1, 0, 1]:
                x1 = x0 + dx
                y1 = y0 + dy
                z1 = z0 + dz

                my_neighbors.append(square_indices[x1, y1, z1])

    neighbors.append(my_neighbors)

neighbors = np.array(neighbors)

def bc_to_3d_array(bc):
    array_3d = np.full((len(bc), 5, 5, 5), -1, dtype=int)
    idx = np.indices(bc._cube_array.shape)[0]
    array_3d[idx, x[np.newaxis], y[np.newaxis], z[np.newaxis]] = bc._cube_array
    return array_3d

def bc_to_3d_bit_array(bc):
    input_array = bc.bit_array().copy()

    input_array = input_array.reshape((-1, 54, 6))
    input_array = np.rollaxis(input_array, 2, 1)

    bit_array_3d = np.full((input_array.shape[0], 6, 5, 5, 5), False, dtype=bool)
    idx, color_idx = np.indices(input_array.shape)[:2]
    bit_array_3d[idx, color_idx, x[np.newaxis, np.newaxis], y[np.newaxis, np.newaxis], z[np.newaxis, np.newaxis]] = input_array
    
    return bit_array_3d



if __name__ == '__main__':
    print("x3d = \\")
    pprint(x)
    print()
    print("y3d = \\")
    pprint(y)
    print()
    print("z3d = \\")
    pprint(z)
    print()

    np.set_printoptions(threshold=np.inf) # allows one to see the whole array even if it is big
    print("neighbors = \\")
    pprint(neighbors)
    print()
    np.set_printoptions(threshold=1000) # allows one to see the whole array even if it is big

    bc = BatchCube(1)
    grid = np.full((5, 5, 5), -1, dtype=int)
    grid[x, y, z] = bc._cube_array[0]
    pprint(grid)

    bc = BatchCube(1)
    bc.randomize(100)
    grid = np.full((5, 5, 5), -1, dtype=int)
    grid[x, y, z] = bc._cube_array[0]
    print()
    print(bc)
    c = np.array(list("rygwob "))
    pprint(c[grid])    

    bc = BatchCube(10)
    grid = np.full((len(bc), 5, 5, 5), -1, dtype=int)
    idx = np.indices(bc._cube_array.shape)[0]
    grid[idx, x[np.newaxis], y[np.newaxis], z[np.newaxis]] = bc._cube_array

    bc = BatchCube(10)
    grid = np.full((len(bc), 5, 5, 5), -1, dtype=int)
    idx = np.indices(bc._cube_array.shape)[0]
    grid[idx, x[np.newaxis], y[np.newaxis], z[np.newaxis]] = bc._cube_array 

    bc = BatchCube(10)
    bc_to_3d_array(bc)
    bc_to_3d_bit_array(bc)

    bc = BatchCube(1)
    pprint(bc_to_3d_bit_array(bc).astype(int))


