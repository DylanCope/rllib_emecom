from .utils import ray_wrapper

from rllib_emecom.train.goal_comms import (
    create_args_parser, get_goal_comms_config
)


def _get_default_args(**override_args):
    parser = create_args_parser()
    args = parser.parse_args([
        item
        for k, v in override_args.items()
        for item in (
            [f'--{k}', str(v)]
            if not isinstance(v, bool)
            else [f'--{k}']
        )
    ])
    return args


@ray_wrapper
def test_train_step():
    args = _get_default_args()
    config = get_goal_comms_config(args)
    algo = config.build()
    algo.train()
