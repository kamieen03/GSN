from enum import Enum, auto


class Mutation(Enum):
    CHANGE_ACTIV = auto()
    INSERT_NODE = auto()
    ADD_CONN = auto()
