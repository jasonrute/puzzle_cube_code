# Solving the Rubic's Cube Using Deep Learning and MCTS

The goal of this project is to use techniques inspired from AlphaGo to solve the Rubic's cube with
little or no domain knowledge.


## Online of the training process:
- (To fill in)

## Implementation Details:
- Master will store the current working project.  
- Good checkpoints will be split off into there own branches.  
- Crazy subprojects (which may break everything) will also be split off.
- Data will be saved in /save but it will be ignored by Git
- NNs will be also by saved in /save but it will be ignored by Git
- A particularly good NN may be saved
- In all cases names will be identifiable with the training run they were generated from
- However, if a 
- A todo list is stored below

## Results (most recent first):
- v0 (2017-10-27)
	- Use MCTS (with policy NN) to generate new policy.  Then train NN with that.
	- Increment max randomization distances (distance from solved cube) at a constant rate
	  and chose a random distance in range [1, max_randomization_distance]
	- No value network (MCTS used value +1 if reached solved state and -1 if reached max_depth)
	- No evalauation of the NN
	- Worked well for low distances, **but at far distances, it wouldn't find the solved state and just memorize the (flawed) policy network.**
	- **In the end, it got pregressively worse.**
  

## To Do
- [x] Dump previous code into Git repository
- [x] Fix Cube action order to make standardized
- [ ] Develop outline of a better training process
  - [ ] Make training file
  - [ ] Split off components of the training process into their own files as it makes sense
- [ ] 