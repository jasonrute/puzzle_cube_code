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
- v0.1.test2 (2017-10-30)
	- First version using training agent
	- New Features:
		- Plays 100 games per generation
		- Game consists of MCTS for each move, then chooses best move as next move
		- If no solution found in MCTS, then quit to save computation
		- Loads best model and saves it
		- Saves a lot of statistics
		- Adaptive difficulty
			- The algorithm isn't great
			- If the win rate for this generation > .85 then increase difficulty +1
			- If the win rate for this generation < .75 then decrease difficulty -1
			- I started with a min distance of 1 and manually changed it half way through to 5
	- Some Parameters
		- cpuct = 10 (turned out to be too big in later stages)
		- max_steps (per MCTS) = 10000 (maybe too many)
		- max_steps (per MCTS) = 10000 (maybe too many)
		- Add in some Dirichlet noise (alpha = .5) to the root node of the MCTS
			- Provides more exploration
			- Prevents the NN from memorizing the prior when it can't find a solution.
	- Results
		- It worked pretty well up to distance 5 and then the MCTS could find the solution,
		  but it often wasn't showing up as the most visited action, so it made the wrong move.
		  (This can be fixed by lowering cpuct to 1.0)
		- The adaptive difficulty algorithm had a bad habit of hanging out at the lowest difficulty.
- v0.0.3 (2017-10-30)
	- Following major changes
		- Add in some Dirichlet noise (alpha = .5) to the root node of the MCTS
			- Provides more exploration
			- Prevents the NN from memorizing the prior when it can't find a solution.
		- Seems to work well, but still doesn't do well above a distance of 7.
		- Also discovered a massive memory leak that needs to be fixed.
- v0.0.2 (2017-10-30)
	- Use basic MCTS as v0 but with the following major changes
		- Count visits before visiting.  Therefore, if I get back to the same node, I won't necessarily loop.
		- *Action values:* Action values are now an estimate of gamma^distance.
		  - Here gamma is .95 and distance is the distance to the solved state.
		  - The leaf values are .01 (about 0.95^90) which help to break the issue where Q is constant 0.
		    The idea is that if we can't reach the solution, then less explored values have a higher Q.
		  - The "max depth" values are 0 (i.e. infinitely far away).
	- It stops working about distcance 7, 8, 9, but unlike v0, it doesn't seem to break at low values.
	- It also doesn't memorize the flawed policy quite as well because of the .01 ambiate value.  (A better approach would be to add some noise to the root node and/or to use the policy from a random symmetric case.)
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
- [x] Generate and save data for early states
- [ ] Fix memory leak (may need to use processes)
- [ ] Improve MCTS algorithm
  - [ ] Add dirichlet noise like in AlphaGo paper (maybe alpha = .75 since there are so few actions)
  - [ ] Record when we get to solution and then set the max_depth to be that value
    - [ ] Not sure what value to use for leaf nodes (maybe 0 is fine)  
- [ ] Develop outline of a better training process
  - [ ] Make training file
  - [ ] Split off components of the training process into their own files as it makes sense
- [ ] 