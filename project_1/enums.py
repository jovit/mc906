from enum import Enum, auto, IntEnum

class Action(Enum):
    GO_FOWARD = auto()
    ROTATE_RIGHT = auto()
    ROTATE_LEFT = auto()
    
class MazePositionType(IntEnum):
    ROBOT = 1
    WALL = 2
    GOAL = 3