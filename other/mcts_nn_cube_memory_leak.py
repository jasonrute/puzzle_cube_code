import numpy as np
from batch_cube import BatchCube
import warnings
import weakref
import gc
import objgraph
from multiprocessing import Process

action_count = 12
c_puct = 10.0
gamma = .95
constant_priors = np.array([1/3] * action_count)
constant_value = .01
max_depth_value = 0.0

class State():
    """ This is application specfic """
    def __init__(self, internal_state=None):
        if internal_state is None:
            internal_state = BatchCube(1)
        
        self.internal_state = internal_state

    def reset_and_randomize(self, depth):
        self.internal_state = BatchCube(1)
        self.internal_state.randomize(depth)

    def next(self, action):
        next_internal_state = self.internal_state.copy()
        next_internal_state.step(action)
        return State(next_internal_state)

    def input_array(self):
        return self.internal_state.bit_array().reshape((1, 6*54))

    def calculate_priors_and_value(self, model):
        """ 
        For now, this does nothing special.  It evenly weights all actions,
        and it gives a nuetral value (0 out of [-1,1]) to each non-terminal node.
        """
        prior = model.predict(self.input_array()).reshape((12,))
        value = .01
        return prior, value

    def key(self):
        return self.internal_state.bit_array().tobytes()

    def done(self):
        return self.internal_state.done()[0]

    def __str__(self):
        return str(self.internal_state)

class MCTSNode():
    def __init__(self, state, mcts_agent):
        self.state = state
        self.terminal = state.done()

        if not self.terminal:
            self.is_leaf_node = True
            self.prior_probabilities, self.node_value = state.calculate_priors_and_value(mcts_agent.model)
            self.total_visit_counts = 0
            self.visit_counts = np.zeros(action_count, dtype=int)
            self.total_action_values = np.zeros(action_count)
            self.mean_action_values = np.zeros(action_count)
            self.children = [None] * action_count

    def upper_confidence_bounds(self):
        return (c_puct * np.sqrt(self.total_visit_counts)) * self.prior_probabilities / (1 + self.visit_counts)

    def child(self, action, mcts_agent):
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
        new_node = MCTSNode(next_state, mcts_agent)
        self.children[action] = new_node
        mcts_agent.transposition_table[key] = new_node
        return new_node

    def select_leaf_and_update(self, max_depth, mcts_agent):
        #print(*(type(o) for o in gc.get_referrers(self)))
        #print("Entering Node:", self.state, "Depth:", max_depth)
        # terminal nodes are good
        if self.terminal:
            #print("... terminal node, returning 1")
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
        action_value = gamma * self.child(action, mcts_agent).select_leaf_and_update(max_depth - 1, mcts_agent)
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
            exponentiated_visit_counts = self.visit_counts ** inv_temp
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

class MCTSAgent():

    def __init__(self, model, initial_state, max_depth, transposition_table={}):
        self.model = model
        self.max_depth = max_depth
        self.total_steps = 0
        self.transposition_table = transposition_table

        self.initial_node = MCTSNode(initial_state, self)
        self.initial_node.prior_probabilities = \
            .75 * self.initial_node.state.calculate_priors_and_value(model)[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]

    def search(self, steps):
        for s in range(steps):
            self.initial_node.select_leaf_and_update(self.max_depth, self) # explore new leaf node
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
        # TOFIX: I should (maybe?) find a way to delete the nodes not below this one, 
        # including from the tranposition table
        best_action = np.argmax(self.action_visit_counts())
        self.initial_node = self.initial_node.child(best_action) 
        self.initial_node.prior_probabilities = \
            .75 * self.initial_node.state.calculate_priors_and_value(model)[0] +\
            .25 * np.random.dirichlet([.5]*action_count, 1)[0]

    def stats(self, key):
        """ Proviods various stats on the MCTS """
        if key == 'to_fill_in_each_keyword':
            #do something
            pass
        else:
            warnings.warn("'{}' argument not implemented for stats".format(key), stacklevel=2)
            return None

## To later remove ##

def prob_box(p):
        return " ▁▂▃▄▅▆▇▉"[int(round(p*8))]
        
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

    #model.load_weights("./save/mcts_nn_cube.h5")
    max_random = 5
    while True:
        max_random += 1
        def f(model, max_random):
            for i in range(100):
                r = 1 + np.random.choice(max_random)
                print()
                print("random dist: {}/{}".format(r, max_random), "step:", i)
                state = State()
                state.reset_and_randomize(r)
                mcts = MCTSAgent(model, state, max_depth=100)
                #print(mcts.initial_node.state)
                if mcts.is_terminal():
                    print("Done!")
                else:
                    mcts.search(steps = 10000)
                    prior, _ = mcts.initial_node.state.calculate_priors_and_value(model)
                    prior2 = mcts.initial_node.prior_probabilities
                    probs = mcts.action_probabilities(inv_temp = 1)
                    q = mcts.initial_node.mean_action_values
                    model.fit(state.input_array(), probs.reshape((1,12)), epochs=1, verbose=0)
                    print("Prior:", "[" + "".join(prob_box(p) for p in prior) + "]")
                    print("PrDir:", "[" + "".join(prob_box(p) for p in prior2) + "]")
                    print("Prob: ", "[" + "".join(prob_box(p) for p in probs) + "]")
                    print("Q:    ", "[" + "".join(prob_box(max(0,p)) for p in q) + "]")

        gc.collect()
        objgraph.show_most_common_types(limit=20)
            
        p = Process(target=f, args=(model, max_random))
        p.start()
        p.join()

        model.save_weights("./save/mcts_nn_cube.h5")

if __name__ == '__main__':
    main()

