import numpy as np
from batch_cube import BatchCube, position_permutations, color_permutations, opp_action_permutations
import warnings

action_count = 12
constant_priors = np.array([1/3] * action_count)
constant_value = .01
max_depth_value = 0.0

def rotationally_randomize(model_policy_value):
    def rotationally_randomized_policy_value(input_array):
        # rotate colors
        input_array = input_array.reshape((54, 6))
        rotation_id = np.random.choice(48)
        pos_perm = position_permutations[rotation_id][:,np.newaxis]
        col_perm = color_permutations[rotation_id][np.newaxis]
        input_array = input_array[pos_perm, col_perm]

        policy, value = model_policy_value(input_array)

        return policy[opp_action_permutations[rotation_id]], value

    return rotationally_randomized_policy_value

class BatchState():
    """ This is application specfic """
    def __init__(self, internal_state=None):
        if internal_state is None:
            internal_state = BatchCube(1)
        
        self.internal_state = internal_state

    def copy(self):
        return State(self.internal_state.copy())

    def import_bit_array(self, bit_array):
        color_idx = np.indices((1, 54, 6))[2]
        array = (color_idx * bit_array.reshape((1, 54, 6))).max(axis=2)
        self.internal_state = BatchCube(cube_array=array)

    def reset_and_randomize(self, depth):
        self.internal_state = BatchCube(1)
        self.internal_state.randomize(depth)

    def next(self, action):
        next_internal_state = self.internal_state.copy()
        next_internal_state.step(action)
        return State(next_internal_state)

    def input_array(self):
        return self.internal_state.bit_array().reshape((1, 6*54))

    def key(self):
        return self.internal_state.bit_array().tobytes()

    def done(self):
        return self.internal_state.done()[0]

    def __str__(self):
        return str(self.internal_state)

class BatchMCTSNodes():
    def __init__(self, mcts_agent, states):
        self.states = states
        self.terminal = states.done()

        if not self.terminal:
            self.c_puct = mcts_agent.c_puct
            self.is_leaf_node = True
            self.prior_probabilities, self.node_value = mcts_agent.model_policy_value(state.input_array())
            self.total_visit_counts = 0
            self.visit_counts = np.zeros(action_count, dtype=int)
            self.total_action_values = np.zeros(action_count)
            self.mean_action_values = np.zeros(action_count)
            self.children = [None] * action_count

    def upper_confidence_bounds(self):
        return (self.node_value * self.c_puct * np.sqrt(self.total_visit_counts)) * self.prior_probabilities / (1 + self.visit_counts)

    def child(self, mcts_agent, action):
        # return node if already indexed
        child_node = self.children[action]
        if child_node is not None:
            return child_node
        
        # check transposition table
        next_state = self.state.next(action)
        key = next_state.key()
        if key in mcts_agent.transposition_table:
            node = mcts_agent.transposition_table[key]
            self.children[action] = node
            return node

        # create new node
        new_node = MCTSNode(mcts_agent, next_state)
        self.children[action] = new_node
        mcts_agent.transposition_table[key] = new_node
        return new_node

    def select_leaf_and_update(self, mcts_agent, max_depth):
        #print("Entering Node:", self.state, "Depth:", max_depth)
        # terminal nodes are good
        if self.terminal:
            #print("... terminal node, returning 1")
            # record shortest distance to target
            depth = mcts_agent.max_depth - max_depth
            if depth < mcts_agent.shortest_path:
                mcts_agent.shortest_path = depth

            return 1.

        # we stop at leaf nodes
        if self.is_leaf_node:
            self.is_leaf_node = False
            #print("... leaf node, returning ", self.node_value)
            return self.node_value

        # reaching max depth is bad
        # (this should punish loops as well)
        if not max_depth:
            #print("... max_depth == 0, returning -1")
            return max_depth_value

        # otherwise, find new action and follow path
        action = np.argmax(self.mean_action_values + self.upper_confidence_bounds())
        
        # update visit counts before recursing in case we come across the same node again
        self.total_visit_counts += 1
        self.visit_counts[action] += 1

        #print("Performing Action:", action)
        action_value = mcts_agent.gamma * \
                       self.child(mcts_agent, action) \
                           .select_leaf_and_update(mcts_agent, max_depth - 1)
        #print("Returning back to:", max_depth)
        
        # recursively update edge values
        self.total_action_values[action] += action_value
        self.mean_action_values[action] = self.total_action_values[action] / self.visit_counts[action]

        # recusively backup leaf value
        #print("DB: update node", "action:", action, "action value:", action_value)
        #print(self.status() + "\n")

        return action_value

    def action_visit_counts(self):
        """ Returns action visit counts. """
        if self.terminal or self.is_leaf_node:
            return np.zeros(action_count, dtype=int)

        return self.visit_counts

    def action_probabilities(self, inv_temp):
        # if no exploring, then this is not defined
        if self.terminal or self.is_leaf_node:
            return None

        if inv_temp == 1:
            return self.visit_counts / self.visit_counts.sum()
        else:
            # scale before exponentiation (the result is the same, but less likely to overflow)
            exponentiated_visit_counts = (self.visit_counts / self.visit_counts.sum()) ** inv_temp
            return exponentiated_visit_counts / exponentiated_visit_counts.sum() 

    def status(self):
        stats = "Node Status:\n"
        stats += str(self.state) + "\n"

        if self.terminal:
            stats += "terminal node"
            return stats

        stats += "Leaf: " + str(self.is_leaf_node) + "\n"
        stats += "Value: " + str(self.node_value) + "\n"
        stats += "Priors: " + str(self.prior_probabilities) + "\n"
        stats += "PUCT: " + str(self.upper_confidence_bounds()) + "\n"
        stats += "Visit Cnts: " + str(self.visit_counts) + "\n"
        stats += "Total Action Values: " + str(self.total_action_values) + "\n"
        stats += "Mean Action Values: " + str(self.mean_action_values)

        return stats

class BatchMCTSNode(initial_states, mcts_agent):
    def __init__():
        # mcts_agent
        self.mcts_agent = mcts_agent

        # counts
        self.evaluated_node_count = 0
        self.allocated_node_count = len(initial_states)
        self.trial_count = len(initial_states)

        # properties per trial id
        self.root_idx = np.arange(self.trial_count) # roots indexed by trial_id
        self.shortest_path = mcts_agent.max_depth + 1

        # allocate initial nodes
        self.states = initial_states
        self.trail_id = np.arange(self.allocated_node_count)
        self.children_idx = np.full(-1, (self.allocated_node_count, 12), dtype=int)
        self.total_visit_counts = np.zeros((self.allocated_node_count, ), dtype=int)
        self.mean_action_values = np.zeros((self.allocated_node_count, ), dtype=float)
        self.total_action_values = np.zeros((self.allocated_node_count, 12), dtype=float)
        self.is_terminal = np.zeros((self.evaluated_node_count, ), dtype=bool)

        # properties per node (created when evaluated)
        self.prior_probabilities = np.zeros((self.evaluated_node_count, ), dtype=float)
        
        self.evaluate_nodes()

    def allocate_nodes(self, trial_idx, states):
        size = len(trial_idx)
        assert size == len(states)
        node_idx = np.arange(self.allocated_node_count, self.allocated_node_count + n)
        self.allocated_node_count += size 

        self.trail_id = np.concatenate(self.trail_id, trial_id)
        self.states = np.concatenate(self.states, states)
        new_children_idx = 
        self.children_idx = \
            np.concatenate(self.children_idx, np.full(-1, (size, 12), dtype=int))
        self.total_visit_counts = \
            np.concatenate(self.total_visit_counts, np.zeros((size, ), dtype=int))
        self.mean_action_values = \
            np.concatenate(self.mean_action_values, np.zeros((size, ), dtype=float))
        self.total_action_values = \
            np.concatenate(self.total_action_values, np.zeros((size, ), dtype=float))
        self.is_terminal = \
            np.concatenate(is_terminal, self.is_terminal(states))

        return node_idx

    def evaluate_nodes(self, node_idx, actions):
        start = self.evaluated_node_count
        end = self.allocated_node_count

        # feed into neural network
        policy, value = self.calc_policy_value(self.states[start:end])
        
        # scale priors for use in the UCB algorithm
        priors = value * self.mcts_agent.c_puct * policy
        self.prior_probabilities = np.concatenate(self.prior_probabilities, priors)

    def upper_confidence_bounds(self, node_idx):
        return (np.sqrt(self.total_visit_counts[node_idx])) * self.prior_probabilities[node_idx] / (1 + self.visit_counts[node_idx])

    def select_leaf_and_update(self, node_idx, remaining_depth):
        action_values = np.zeros(node_idx.shape[0])

        # if at the bottom of the recursion, then evaluate all the leaf nodes at the same time
        if not len(node_idx) or not remaining_depth:
            self.evaluate_leaf_nodes()
            action_values = self.mcts_agent.max_depth_value # reaching max depth is bad
            return action_values

        # terminal nodes are good (also record shortest path)
        is_terminal = self.is_terminal[node_idx]
        terminal_idx = node_idx[terminal]
        action_values[is_terminal] == 1
        terminal_trail_id = self.trial_id[terminal_idx]
        self.shortest_path[terminal_trail_id] = \
            min(self.shortest_path[terminal_trail_id], self.mcts_agent.max_depth - remaining_depth)

        # handle leaf nodes after recursion
        is_leaf = (node_idx >= self.evaluated_node_count)
        leaf_idx = node_idx[is_leaf]

        # otherwise, find new action and follow path
        is_intermediate = (~is_terminal) & (~is_leaf)
        intermediate_idx = node_idx[intermediate]

        actions = np.argmax(self.mean_action_values[intermediate_idx] 
                            + self.upper_confidence_bounds(intermediate_idx), axis=1)
        
        # update visit counts before recursing in case we come across the same node again
        self.total_visit_counts[intermediate_idx] += 1
        self.visit_counts[intermediate_idx, actions] += 1

        # create children to recurse on
        child_node_idx = self.find_children(intermediate_idx, actions)

        action_values[is_intermediate] = self.mcts_agent.gamma * \
                                         self.select_leaf_and_update(child_node_idx, max_depth - 1)

        # recursively update edge values
        self.total_action_values[intermediate_idx, actions] += action_values[is_intermediate]
        self.mean_action_values[intermediate_idx, actions] = \
            self.total_action_values[intermediate_idx, action] / self.visit_counts[intermediate_idx, action]

        # handle leaf nodes (the leaf node evaluation is done at the bottom of the recursion)
        action_values[is_leaf] = self.node_value[leaf_idx]

        # recusively backup leaf value
        return action_values

    def transposition_table_table_look_up(self, trial_idx, states):
        for trial_id, state in zip(trial_idx, states):
            if (trial_id, key) in self.mcts_agent.transposition_table:
                yield self.mcts_agent.transposition_table[trial_id, key]
            else:
                yield -1

    def transposition_table_update(self, trial_idx, node_idx):
        for trial_id, node_id in zip(trial_idx, states):
            self.mcts_agent.transposition_table[trial_id, key] = node_id

    def find_children(self, node_idx, actions):
        # check if created
        child_node_idx = self.children[node_idx, action]
        has_new_child = (child_node_idx == -1)
        has_new_child_idx = node_idx[has_new_child]

        # find next state
        next_states = self.next_state(has_new_child_idx, actions)
        trial_idx = self.trial_id[has_new_child_idx], 
        keys = self.get_keys(next_states)
         
        # look in transposition table
        child_node_idx[has_new_child_idx] = self.transposition_table_lookup(trial_idx, keys)
        
        # handle ones not found
        still_has_new_child = (child_node_idx[has_new_child_idx] == -1)
        next_states = next_states[still_has_new_child]
        keys = keys[still_has_new_child]

        has_new_child = (child_node_idx == -1)
        has_new_child_idx = node_idx[has_new_child]

        # create node if needed
        child_node_idx[has_new_child] = self.allocate_nodes(self.trial_id[has_new_child_idx], next_states)

        # update table
        self.transposition_table_update(trial_idx, keys, child_node_idx[has_new_child])

        # update children
        self.children[node_idx, action] = child_node_idx

        return child_node_idx

    def next_state(self, node_idx, actions):
        batch_cube = BatchCube(cube_array = self.states[node_idx])
        batch_cube.step(actions)
        states = batch_cube._cube_array

        return states

class BatchMCTSAgent():

    def __init__(self, model_policy_value, initial_states, max_depth, transposition_table={}, c_puct=1.0, gamma=.95):
        self.model_policy_value = rotationally_randomize(model_policy_value)
        self.max_depth = max_depth
        self.total_steps = 0
        self.transposition_table = transposition_table
        self.c_puct = c_puct  # exploration constant
        self.gamma = gamma  # decay constant

        self.batch_nodes = BatchMCTSNode(self, initial_states)

        self.initial_node.prior_probabilities = \
            .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]

        self.shortest_path = self.max_depth + 1

    def search(self, steps):
        for s in range(steps):
            root_idx = self.initial_node_idx
            self.select_leaf_and_update(self.initial_node_idx, root_idx, self.max_depth)
            self.total_steps += 1

    def action_visit_counts(self):
        return self.initial_node.action_visit_counts()
    
    def action_probabilities(self, inv_temp):
        return self.initial_node.action_probabilities(inv_temp)

    def initial_node_status(self):
        return self.initial_node.status()

    def is_terminal(self):
        return self.initial_node.terminal

    def advance_to_best_child(self):
        """ Advance to the best child node """
        
        best_action = np.argmax(self.action_visit_counts())
        self.advance_to_action(best_action)

    def advance_to_action(self, action):
        """ Advance to a child node via the given action """
        
        # TOFIX: I should (maybe?) find a way to delete the nodes not below this one, 
        # including from the tranposition table
        
        self.initial_node = self.initial_node.child(self, action) 
        self.initial_node.prior_probabilities = \
            .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]
        
        self.shortest_path = self.max_depth + 1

    def stats(self, key):
        """ Proviods various stats on the MCTS """
        
        if key == 'shortest_path':
            return self.shortest_path if self.shortest_path <= self.max_depth else -1
        elif key == 'prior':
            return self.model_policy_value(self.initial_node.state.input_array())[0]
        elif key == 'prior_dirichlet':
            return self.initial_node.prior_probabilities
        elif key == 'value':
            return self.model_policy_value(self.initial_node.state.input_array())[1]
        elif key == 'visit_counts':
            return self.initial_node.visit_counts
        elif key == 'total_action_values':
            return self.initial_node.total_action_values
        else:
            warnings.warn("'{}' argument not implemented for stats".format(key), stacklevel=2)
            return None


class BatchMCTSAgent():

    def __init__(self, model_policy_value, initial_states, max_depth, transposition_table={}, c_puct=1.0, gamma=.95):
        self.model_policy_value = rotationally_randomize(model_policy_value)
        self.max_depth = max_depth
        self.total_steps = 0
        self.transposition_table = transposition_table
        self.c_puct = c_puct  # exploration constant
        self.gamma = gamma  # decay constant

        self.initial_node = MCTSNode(self, initial_state)
        self.initial_node.prior_probabilities = \
            .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]

        self.shortest_path = self.max_depth + 1

    def search(self, steps):
        for s in range(steps):
            self.initial_node.select_leaf_and_update(self, self.max_depth) # explore new leaf node
            self.total_steps += 1

    def action_visit_counts(self):
        return self.initial_node.action_visit_counts()
    
    def action_probabilities(self, inv_temp):
        return self.initial_node.action_probabilities(inv_temp)

    def initial_node_status(self):
        return self.initial_node.status()

    def is_terminal(self):
        return self.initial_node.terminal

    def advance_to_best_child(self):
        """ Advance to the best child node """
        
        best_action = np.argmax(self.action_visit_counts())
        self.advance_to_action(best_action)

    def advance_to_action(self, action):
        """ Advance to a child node via the given action """
        
        # TOFIX: I should (maybe?) find a way to delete the nodes not below this one, 
        # including from the tranposition table
        
        self.initial_node = self.initial_node.child(self, action) 
        self.initial_node.prior_probabilities = \
            .75 * self.model_policy_value(self.initial_node.state.input_array())[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]
        
        self.shortest_path = self.max_depth + 1

    def stats(self, key):
        """ Proviods various stats on the MCTS """
        
        if key == 'shortest_path':
            return self.shortest_path if self.shortest_path <= self.max_depth else -1
        elif key == 'prior':
            return self.model_policy_value(self.initial_node.state.input_array())[0]
        elif key == 'prior_dirichlet':
            return self.initial_node.prior_probabilities
        elif key == 'value':
            return self.model_policy_value(self.initial_node.state.input_array())[1]
        elif key == 'visit_counts':
            return self.initial_node.visit_counts
        elif key == 'total_action_values':
            return self.initial_node.total_action_values
        else:
            warnings.warn("'{}' argument not implemented for stats".format(key), stacklevel=2)
            return None

## To later remove ##

def prob_box(p):
        return " ▁▂▃▄▅▆▇█"[int(round(p*8))]
        
def main():
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.optimizers import Adam
    from keras.losses import categorical_crossentropy

    model = Sequential()
    model.add(Dense(128, input_dim=6*54, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(12, activation='softmax'))
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001))

    def model_policy_value(input_array):
        policy = model.predict(input_array).reshape((12,))
        value = 0.1
        return policy, value

    #model.load_weights("./save/mcts_nn_cube.h5")
    max_random = 5
    while True:
        max_random += 1
        for i in range(100):
            r = 1 + np.random.choice(max_random)
            print()
            print("random dist: {}/{}".format(r, max_random), "step:", i)
            state = State()
            state.reset_and_randomize(r)
            mcts = MCTSAgent(model_policy_value, state, max_depth=100)
            #print(mcts.initial_node.state)
            if mcts.is_terminal():
                print("Done!")
            else:
                mcts.search(steps = 10000)
                prior, _ = model_policy_value(mcts.initial_node.state.input_array())
                prior2 = mcts.initial_node.prior_probabilities
                probs = mcts.action_probabilities(inv_temp = 1)
                q = mcts.initial_node.mean_action_values
                model.fit(state.input_array(), probs.reshape((1,12)), epochs=1, verbose=0)
                print("Prior:", "[" + "".join(prob_box(p) for p in prior) + "]")
                print("PrDir:", "[" + "".join(prob_box(p) for p in prior2) + "]")
                print("Prob: ", "[" + "".join(prob_box(p) for p in probs) + "]")
                print("Q:    ", "[" + "".join(prob_box(max(0,p)) for p in q) + "]")

        model.save_weights("./save/mcts_nn_cube.h5")

if __name__ == '__main__':
    main()

