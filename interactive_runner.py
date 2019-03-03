"""
Interactive cube enviroment
"""

import batch_cube
import numpy as np

valid_moves = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]

help_string = """
Interactive cube solver.

Commands:
  reset
    Go back to solved cube (without solving it).
  solve
    Attempts to solve cube.  Makes move after finding a solution or 
    exploring 10000 nodes in search tree.
  solve <nodes>
    Attempts to solve cube.  Manually set a positive number of nodes to explore.
    Example: solve 10000
  scramble <distance>
    Scrambles cube a fixed number of moves (distance must be positive).
    Example: scamble 15
  move <moves>
    Manually move cube one or more actions seperated by spaces.
    Valid moves are L R U D F B or one of those moves followed by 
    ' to twist in the opposite direction (e.g. L').
    Example: move L U' L' U F
  help
    Print this help text again.
  quit
    Terminate the program.
  
"""

class CubeState:
    def __init__(self):
        self.cube = batch_cube.BatchCube()
    
    def reset(self):
        self.cube = batch_cube.BatchCube()

    def solve(self, nodes = 10000):
        print("NOT IMPLEMENTED YET")

    def scramble(self, distance):
        self.cube.randomize(distance)
    
    def move(self, moves):
        for m in moves:
            i = valid_moves.index(m)
            self.cube.step(np.array([i]))
    
if __name__ == "__main__":
    cube_state = CubeState()

    print(help_string)

    while True:
        print(cube_state.cube)
        
        args = input("> ").split()
        if args:
            if args[0] == 'reset':
                if len(args) == 1:
                    cube_state.reset()
                else:
                    print("Invalid solve command. Type 'help' for help.")
            elif args[0] == 'solve':
                if len(args) == 1:
                    cube_state.solve()
                elif len(args) == 2 and args[1].isdigit() and int(args[1]) > 0:
                    node_limit = int(args[1])
                    cube_state.solve(node_limit)
                else:
                    print("Invalid solve command. Type 'help' for help.")
            elif args[0] == 'scramble':
                if len(args) == 2 and args[1].isdigit() and int(args[1]) > 0:
                    distance = int(args[1])
                    cube_state.scramble(distance)
                else:
                    print("Invalid scramble command. Type 'help' for help.")
            elif args[0] == 'move':
                if len(args) >= 2 and all((x in valid_moves) for x in args[1:]):
                    moves = args[1:]
                    cube_state.move(moves)
                else:
                    print("Invalid move command. Type 'help' for help.")
            elif args[0] == 'help':
                print(help_string)
            elif args[0] == 'quit':
                print("Good bye!")
                break
            else:
                print("Invalid command.  Type 'help' for options or 'quit' to exit.")

        
