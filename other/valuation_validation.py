from typing import Optional, List

from cube_model import CubeModel
from mcts_nn_cube import State
from puzzle_cube import PuzzleCube


class CubeValueSolver:

    def __init__(self, cube: PuzzleCube, model: CubeModel):
        """
        :param cube: The starting puzzle cube.
        :param model: The trained model to use.
        """

        assert (model._model is not None), "model must be loaded"
        history_length = model._model.history
        blank_history = tuple(None for _ in range(history_length - 1))
        internal_state = (cube._inner_cube,) + blank_history
        initial_state = State(_internal_state=internal_state)

        self.model_policy_value = model._function()
        self.state = initial_state  # type: State
        self.solution_length = 0

    def solve(self, steps: int) -> None:
        """
        Run the solver for a certain number of steps.
        :param steps: Number of steps to run the valuation solver.
        (More steps may find shorter solutions.)  Default is True.
        """
        for i in range(steps):
            best_child = None  # type: Optional[State]
            best_value = 0.0
            for a in range(12):
                c = self.state.next(a)
                if c.done():
                    value = 1.0
                else:
                    value = self.model_policy_value(c.input_array())[1]
                if value > best_value:
                    best_child = c
                    best_value = value

            self.state = best_child
            self.solution_length += 1
            if self.state.done():
                break

    def solution(self) -> Optional[List[str]]:
        if self.state.done():
            return ["Solution Found", self.solution_length]

        return None


if __name__ == '__main__':
    from collections import Counter

    cnts = Counter()
    model_file_path = "../example/checkpoint_model_v1.0.5-r_gen034.h5"
    for d in range(1, 30):
        cnt_solved = 0
        for _ in range(100):
            pc = PuzzleCube()
            pc = pc.scramble(d)

            cm = CubeModel()
            cm.load_from_config(model_file_path)

            solver = CubeValueSolver(pc, cm)
            solver.solve(steps=50)

            if solver.solution():
                cnt_solved += 1

        cnts[d] = cnt_solved
        print("Distance:", d, "Solved:", cnt_solved, "/ 100")
        if cnt_solved == 0:
            break

    print(cnts)
