from enum import Enum, auto, IntEnum

class Action(Enum):
    GO_RIGHT = auto()
    GO_LEFT = auto()
    GO_UP = auto()
    GO_DOWN = auto()
    
class MazePositionType(IntEnum):
    ROBOT = auto()
    WALL = auto()
    GOAL = auto()
    VISITED = auto()
    PATH = auto()
    EMPTY = auto()


color_map = {
        MazePositionType.EMPTY: [255, 255, 255, 0],
        MazePositionType.WALL: [0, 0, 0, 255],
        MazePositionType.PATH: [24, 144, 136, 255],
        MazePositionType.ROBOT: [255, 0, 26, 255],
        MazePositionType.VISITED: [209, 187, 161, 150],
        MazePositionType.GOAL: [0, 255, 89, 255],
    }