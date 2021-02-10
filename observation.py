from enum import Enum, auto


class Observation(Enum):
    Prior = auto()

    Resource = auto()
    ResourceUp = auto()
    ResourceDown = auto()
    ResourceLeft = auto()
    ResourceRight = auto()

    AgentUp = auto()
    AgentDown = auto()
    AgentLeft = auto()
    AgentRight = auto()

    OutUp = auto()
    OutDown = auto()
    OutLeft = auto()
    OutRight = auto()
