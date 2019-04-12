from search import *
from helpers import maze_generator, plot_tile_map
from enums import MazePositionType
from RobotProblem import RobotProblem
from copy import copy, deepcopy
import numpy as np

maze = maze_generator()

robot_problem = RobotProblem(maze)

breadth = breadth_first_graph_search(RobotProblem(maze))
depth_path = depth_first_graph_search(RobotProblem(maze))
#iterative_deepening = iterative_deepening_search(RobotProblem(maze))

breadth_maze = maze.copy()
for d in breadth.path():
    s = d.state
    if breadth_maze[s[0], s[1]] != MazePositionType.ROBOT and breadth_maze[s[0], s[1]] != MazePositionType.GOAL:
        breadth_maze[s[0], s[1]] = MazePositionType.PATH
    print(s)
plot_tile_map(breadth_maze)

deapth_maze = maze.copy()
for d in depth_path.path():
    s = d.state
    if deapth_maze[s[0], s[1]] == MazePositionType.PATH:
        print("igual!")
    
    if deapth_maze[s[0], s[1]] != MazePositionType.ROBOT and deapth_maze[s[0], s[1]] != MazePositionType.GOAL:
        deapth_maze[s[0], s[1]] = MazePositionType.PATH
    print(s)
plot_tile_map(deapth_maze)

print(maze)

# for d in iterative_deepening.path():
#     s = d.state
#     print(s)