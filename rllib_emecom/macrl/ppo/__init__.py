from .macrl_ppo import PPOMACRL
from ray.tune.registry import register_trainable


register_trainable('PPOMACRL', PPOMACRL)
