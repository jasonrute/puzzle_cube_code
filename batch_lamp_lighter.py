import numpy as np

basic_actions = "LRA" #left, right, toggle lamp

WIDTH = 8
empty_lamp = np.zeros(WIDTH, dtype=bool)

blank_switches = np.zeros((WIDTH, WIDTH), dtype=bool)
light_switches = np.eye(WIDTH, dtype=bool)
switches = np.array([blank_switches, blank_switches, light_switches])
moves = np.array([-1, 1, 0])

#symbol_map = {(False, False): "│", (False, True): "┃", (True, False): "┼", (True, True): "╂"}
#symbol_map = {(False, False): "└", (False, True): "╙", (True, False): "┴", (True, True): "╨"}  
#symbol_map = {(False, False): " ", (False, True): ":", (True, False): "-", (True, True): "+"}  
#symbol_map = {(False, False): "┬", (False, True): "┴", (True, False): "╦", (True, True): "╩"}
#symbol_map = {(False, False): "↓", (False, True): "↑", (True, False): "⇓", (True, True): "⇑"}
symbol_map = {(False, False): " - ", (False, True): " + ", (True, False): "(-)", (True, True): "(+)"}

class BatchLampLighter():
    """
    An implementation of a vector of Rubik's cubes using NumPy arrays.
    It has methods for converting to the PyCuber object class.
    """
    
    def __init__(self, length = 1):
        """
        Creates length-many solved lamps
        """
        self._light_array = np.repeat(empty_lamp[np.newaxis], repeats=length, axis=0)
        self._position_array = np.zeros(length, dtype=int)
    
    def copy(self):
        new_ll = BatchLampLighter(len(self))
        new_ll._light_array = self._light_array.copy()
        new_ll._position_array = self._position_array.copy()

        return new_ll

    def __len__(self):
        return self._light_array.shape[0]
     
    def bit_array(self):
        position_bit_array = np.eye(WIDTH, dtype=bool)[self._position_array]
        return np.concatenate([self._light_array, position_bit_array], axis=1)

    def randomized_state(self):
        random_shifts = np.random.choice(WIDTH, len(self))
        new_position_array = (self._position_array + random_shifts) % WIDTH
        random_shifts_bit_array = np.eye(WIDTH, dtype=bool)[random_shifts]
        new_position_bit_array = np.eye(WIDTH, dtype=bool)[new_position_array]

        sample_index, row_index = np.indices(self._light_array.shape)
        row_index -= random_shifts[:, np.newaxis]
        row_index %= WIDTH
        random_switches = np.random.choice(2, (len(self), WIDTH)).astype(bool)
        new_light_array = self._light_array[sample_index, row_index] ^ random_switches

        return np.concatenate([random_shifts_bit_array, random_switches, new_position_bit_array, new_light_array], axis=1)

    def step(self, actions):
        """
        Assuming actions is a list of length = len(self)
        """
        action_moves = moves[actions]
        action_switches = switches[actions, self._position_array]
        sample_index, _ = np.indices(self._light_array.shape)
        self._light_array ^= action_switches
        self._position_array += action_moves
        self._position_array %= WIDTH

    def step_independent(self, actions):
        """
        Performs all actions independently on each state
        """
        action_len = len(actions)
        sample_len = len(self._light_array)
        
        self._light_array = np.repeat(self._light_array, repeats=action_len, axis=0)
        self._position_array = np.repeat(self._position_array, repeats=action_len, axis=0)
        actions = np.tile(actions, sample_len)
        
        self.step(actions)

    def randomize(self, dist=100):
        l = len(self._light_array)
        for _ in range(dist):
            actions = np.random.choice(3, l)
            self.step(actions)
    
    def done(self):
        return (self._light_array == empty_lamp).all(axis=1) & (self._position_array == 0)
    
    def remove_done(self):
        done = self.done()
        self._light_array = self._light_array[~done]
        self._position_array = self._position_array[~done]

    def __str__(self):
        out = []
        for lights, position in zip(self._light_array, self._position_array):
            line = "".join(symbol_map[n == position, light] for n, light in enumerate(lights))
            out.append(line)

        return "\n".join(out)

if __name__ == '__main__':
    # test all actions
    for i, m in enumerate(basic_actions):
        bc = BatchLampLighter()
        
        bc.step(np.array([i]))
        
        print(bc)
    
    print()

    # multiple actions
    bc = BatchLampLighter(3)
    bc.step([2,2,2]) 
    
    bc.step(np.arange(3)) 
    
    print(bc)
    
    print()

    # test remove_done
    bc = BatchLampLighter(1)
    
    bc.step_independent(np.arange(3))
    bc.step_independent(np.arange(3))   
    print(bc) 
    print(bc.done())
    bc.remove_done()
    print(len(bc))

    # test independent actions
    bc.step_independent(np.arange(3))
    bc.step_independent(np.arange(3))
    print(len(bc))
    print(bc.done().sum())

    #test randomize
    bc = BatchLampLighter(2)
    bc.randomize(1)
    print(bc)

    bc.randomize(10)
    print(bc)

    bc.bit_array().shape

    #test random state
    print()

    bc = BatchLampLighter(1)
    bc.step(2)
    bc.step(1)
    print(bc)
    state = bc.randomized_state()
    for line in state.astype(int).reshape(4, WIDTH):
        print("".join(map(str, line)))

    


    