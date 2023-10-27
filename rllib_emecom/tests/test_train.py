from .utils import ray_wrapper, parse_default_args

from rllib_emecom.train.goal_comms import (
    create_goal_comms_args_parser, get_goal_comms_config
)


@ray_wrapper
def test_train_step():
    parser = create_goal_comms_args_parser()
    args = parse_default_args(parser)
    config = get_goal_comms_config(args)
    algo = config.build()
    algo.train()
