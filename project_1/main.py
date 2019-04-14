import sys
sys.path.insert(0,'/Users/joao.goncalves/Documents/Repos/aima-python')
from search import *
from helpers import maze_generator, plot_tile_map
from enums import MazePositionType
from RobotProblem import RobotProblem
from copy import copy, deepcopy
import numpy as np

def plot_path(maze, path):
    m = maze.copy()
    for d in path.path():
        s = d.state        
        if m[s[0], s[1]] != MazePositionType.ROBOT and m[s[0], s[1]] != MazePositionType.GOAL:
            m[s[0], s[1]] = MazePositionType.PATH
    for d in path.explored():
        s = d.state        
        if m[s[0], s[1]] != MazePositionType.ROBOT and m[s[0], s[1]] != MazePositionType.GOAL:
            m[s[0], s[1]] = MazePositionType.PATH
    plot_tile_map(m)

maze = maze_generator()

robot_problem = RobotProblem(maze)

breadth = breadth_first_graph_search(RobotProblem(maze))
depth_path = depth_first_graph_search(RobotProblem(maze))
problem = RobotProblem(maze)
a_star_sqr_path = astar_search(RobotProblem(maze), h=memoize(problem.squared_manhattan_distance, "squared_manhattan_distance"))
problem = RobotProblem(maze)
a_star_distance_path = astar_search(RobotProblem(maze), h=memoize(problem.manhattan_distance, "manhattan_distance"))
#iterative_deepening = iterative_deepening_search(RobotProblem(maze))

plot_path(maze, breadth)

plot_path(maze, depth_path)


plot_path(maze, a_star_sqr_path)

plot_path(maze, a_star_distance_path)


# for d in iterative_deepening.path():
#     s = d.state
#     print(s)