from .q_learner import QLearner
from .coma_learner import COMALearner
from .rode_learner import RODELearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["rode_learner"] = RODELearner