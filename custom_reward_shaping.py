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

def custom_reward(env, state, action):
  # calculate the action the heuristic would choose
  heuristic_action = heuristic(env, state)
  
  # calculate the difference between the algorithm action and the heuristic action
  # currently this is just a naive method, where the reward is the total negative
  # absolute value difference between the two actions.
  # We might look into more sophisticated methods if time allows
  return -1.0 * (abs(heuristic_action[0] - action[0]) + abs(heuristic_action[1] - action[1]))
  
  
# heuristic function copied from https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
# not sure if changes will be needed
def heuristic(env, s):
  angle_targ = s[0] * 0.5 + s[2] * 1.0
  if angle_targ > 0.4:
      angle_targ = 0.4
  if angle_targ < -0.4:
      angle_targ = -0.4
  hover_targ = 0.55 * np.abs(
      s[0]
  )

  angle_todo = (angle_targ - s[4]) * 0.5 - (s[5]) * 1.0
  hover_todo = (hover_targ - s[1]) * 0.5 - (s[3]) * 0.5

  if s[6] or s[7]:
      angle_todo = 0
      hover_todo = (
          -(s[3]) * 0.5
      )

  if env.continuous:
      a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
      a = np.clip(a, -1, +1)
  else:
      a = 0
      if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
          a = 2
      elif angle_todo < -0.05:
          a = 3
      elif angle_todo > +0.05:
          a = 1
  return a
