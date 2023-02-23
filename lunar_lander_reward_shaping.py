# Custom reward function to use in reward shaping
# Calculates a "reward" value given a state and action
# based on the action and the heuristic action compare

# PARAMETERS
# env: the environment
# state: current state of the environment and agent
# action: calculated next action based on the <TODO> algorithm

# RETURNS:
# Reward value that represents how different the given action
# and the action calculated by the heuristic are

# Currently this value will be non-positive, with 0 being
# that the two actions are exactly the same, and gets more
# negative the more different they are.
import numpy as np

def calc_shaping_rewards(state, action):
	"""
	Input: 
	 - state: (batch size)x(8) tensor with current observations
	 - action: (batch_size)x(2) tensor with current action

	Output:
	 - rewards: (batch_size)x(1) tensor with shaping rewards for each (s,a) pair in the batch.
	"""
	# calculate the action the heuristic would choose, based on https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
	angle_targ = state[0] * 0.5 + state[2] * 1.0
	if angle_targ > 0.4:
		angle_targ = 0.4
	if angle_targ < -0.4:
		angle_targ = -0.4
	hover_targ = 0.55 * np.abs(state[0])

	angle_todo = (angle_targ - state[4]) * 0.5 - (state[5]) * 1.0
	hover_todo = (hover_targ - state[1]) * 0.5 - (state[3]) * 0.5

	if state[6] or state[7]:
		angle_todo = 0
		hover_todo = (-(state[3]) * 0.5)

	heuristic_action = np.array([hover_todo * 20 - 1, -angle_todo * 20])
	heuristic_action = np.clip(heuristic_action, -1, +1)
	
	# calculate the difference between the algorithm action and the heuristic action
	# currently this is just a naive method, where the reward is the total negative
	# absolute value difference between the two actions.
	# We might look into more sophisticated methods if time allows
	return -1.0 * (abs(heuristic_action[0] - action[0]) + abs(heuristic_action[1] - action[1]))
