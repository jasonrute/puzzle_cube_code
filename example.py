from cube_model import CubeModel
from cube_solver import CubeSolver
from puzzle_cube import PuzzleCube

# load the neural network from a file of saved weights
cm = CubeModel()
cm.load_from_config("../example/checkpoint_model_v1.0.5-r_gen034.h5")

# create a new puzzle cube and randomize it
pc = PuzzleCube()
pc = pc.scramble(8)
print(pc)

# use Monte Carlo tree search with the loaded neural network to solve the cube
s = CubeSolver(pc, cm)
s.solve(steps=1600)
print(s.solution())

# verify that this solution works
for action in s.solution():
    pc = pc.move(action)
assert pc.is_solved()