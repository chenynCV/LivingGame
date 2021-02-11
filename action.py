from enum import Enum, auto


class Action(Enum):
    Nothing = auto()

    MoveUp = auto()
    MoveDown = auto()
    MoveLeft = auto()
    MoveRight = auto()

    Create = auto()
