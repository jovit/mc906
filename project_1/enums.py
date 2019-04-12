from enum import Enum, auto, IntEnum

class Action(Enum):
    GO_FOWARD = auto()
    ROTATE_RIGHT = auto()
    ROTATE_LEFT = auto()
    
class MazePositionType(IntEnum):
    ROBOT = auto()
    WALL = auto()
    GOAL = auto()
    VISITED = auto()
    PATH = auto()
    EMPTY = auto()


color_map = {
        MazePositionType.EMPTY: [255, 255, 255, 0],
        MazePositionType.WALL: [65, 63, 62, 255],
        MazePositionType.PATH: [24, 144, 136, 255],
        MazePositionType.ROBOT: [117, 151, 143, 255],
        MazePositionType.VISITED: [209, 187, 161, 150],
        MazePositionType.GOAL: [235, 101, 89, 255],
    }