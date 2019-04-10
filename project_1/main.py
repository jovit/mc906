from search import *
from helpers import maze_generator
from RobotProblem import RobotProblem

maze = maze_generator()

robot_problem = RobotProblem(maze)

breadth = breadth_first_graph_search(RobotProblem(maze))
depth_path = depth_first_graph_search(RobotProblem(maze))
#iterative_deepening = iterative_deepening_search(RobotProblem(maze))

for d in breadth.path():
    s = d.state
    print(s)

for d in depth_path.path():
    s = d.state
    print(s)

# for d in iterative_deepening.path():
#     s = d.state
#     print(s)