from enums import MazePositionType
import numpy as np
import random

def maze_generator(): 
    maze = np.zeros((25, 25))
    
    maze[0][0] = MazePositionType.ROBOT # indicates the current position of the robot
    
    current_line = 0
    for line in maze:
        if current_line % 5 == 0 and current_line != 0:
            if bool(random.getrandbits(1)):
                line[10:] = MazePositionType.WALL
            else:
                line[:-10] = MazePositionType.WALL
        current_line = current_line + 1
        
    maze[24][24] = MazePositionType.GOAL # indicates the goal
    return maze


def does_visited_states_contain(visited_states, state):
    for s in visited_states:
        if s["x"] == state["x"] and s["y"] == state["y"] and s["angle"] == state["angle"]:
            return True
    return False