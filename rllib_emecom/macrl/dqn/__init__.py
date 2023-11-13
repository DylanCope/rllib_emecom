from .macrl_dqn import DQNMACRL
from ray.tune.registry import register_trainable


register_trainable('DQNMACRL', DQNMACRL)
