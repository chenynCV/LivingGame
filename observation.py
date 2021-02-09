from enum import Enum, auto


class Observation(Enum):
    Resources = auto()

    AgentUp = auto()
    AgentDown = auto()
    AgentLeft = auto()
    AgentRight = auto()

    OutUp = auto()
    OutDown = auto()
    OutLeft = auto()
    OutRight = auto()
