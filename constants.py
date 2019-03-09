
MOVE_AHEAD = 'MoveAhead'
ROTATE_LEFT = 'RotateLeft'
ROTATE_RIGHT = 'RotateRight'
LOOK_UP = 'LookUp'
LOOK_DOWN = 'LookDown'
DONE = 'Done'
FIND_MORE = 'FindMore'  # Organick modified

#BASIC_ACTIONS = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE]
BASIC_ACTIONS = [MOVE_AHEAD, ROTATE_LEFT, ROTATE_RIGHT, LOOK_UP, LOOK_DOWN, DONE, FIND_MORE]  # Organick modified

INTERMEDIATE_REWARD = 3  # Organick modified
GOAL_SUCCESS_REWARD = 5
STEP_PENALTY = -0.03
FAILED_ACTION_PENALTY = 0