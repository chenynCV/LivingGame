from enum import Enum, auto


class Observation(Enum):
    Nothing = auto()
    Resources = auto()

    UpAgent = auto()
    DownAgent = auto()
    LeftAgent = auto()
    RightAgent = auto()

    UpOut = auto()
    DownOut = auto()
    LeftOut = auto()
    RightOut = auto()
