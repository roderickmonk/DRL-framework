from enum import Enum, auto


class EpsilonDecayType(Enum):
    LINEAR = auto()
    EXPONENTIAL = auto()


class ActionSpace:
    Discrete = auto()
    Continuous = auto()
