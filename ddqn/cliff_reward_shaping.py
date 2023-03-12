def calc_shaping_rewards(state, action):
    top_row_states = [0,1,2,3,4,5,6,7,8,9,10,11]
    if state in top_row_states and action == 1:
        pass