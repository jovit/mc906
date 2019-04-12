import sys

from search import *
import numpy as np
from enums import MazePositionType, Action
from helpers import maze_generator, does_visited_states_contain

class RobotProblem(Problem):

    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, maze):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        x_robot, y_robot = np.where(maze == MazePositionType.ROBOT)
        x_goal, y_goal = np.where(maze == MazePositionType.GOAL)
        initial_state = (x_robot[0], y_robot[0], 0)

        Problem.__init__(self, initial_state, (x_goal[0], y_goal[0]))

        self.maze = maze

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        actions = []
        

        actions.append(Action.ROTATE_LEFT)


        actions.append(Action.ROTATE_LEFT)

        if (state[2] == 0 and state[1] + 1 < len(self.maze[state[0]]) and self.maze[state[0], state[1] + 1] != MazePositionType.WALL) or \
           (state[2] == 90 and state[0] + 1 < len(self.maze) and self.maze[state[0] + 1, state[1]] != MazePositionType.WALL) or \
           (state[2] == 180 and state[1] - 1 >= 0 and self.maze[state[0], state[1] - 1] != MazePositionType.WALL) or \
           (state[2] == 270 and state[0] - 1 >= 0 and self.maze[state[0] - 1, state[1]] != MazePositionType.WALL):
                actions.append(Action.GO_FOWARD)
        return actions



    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        x = state[0]
        y = state[1]
        angle = state[2]
        
        if action == Action.ROTATE_LEFT:
            angle -= 90
            if angle < 0: 
                angle = 270
        elif action == Action.ROTATE_RIGHT:
            angle += 90
            if angle == 360: 
                angle = 0
        elif action == Action.GO_FOWARD:
            if angle == 0:
                y += 1
            elif angle == 90:
                x += 1
            elif angle == 180:
                y -= 1
            elif angle == 270:
                x -= 1
        return (x, y, angle)

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return self.goal[0] == state[0] and self.goal[1] == state[1]

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2.  If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value.  Hill-climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError