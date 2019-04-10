from search import *
from helpers import maze_generator
from RobotProblem import RobotProblem

maze = maze_generator()

robot_problem = RobotProblem(maze)

#breadth = breadth_first_tree_search(RobotProblem(maze))
depth_path = depth_first_tree_search(RobotProblem(maze)).path()
#interative_deepening = iterative_deepening_search(RobotProblem(maze))

for d in depth_path:
    s = d.state
    print(dict(x=s["x"], y=s["y"], angle=s["angle"]))
