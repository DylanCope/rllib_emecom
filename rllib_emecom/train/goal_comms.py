from rllib_emecom.utils.experiment_utils import get_wandb_callback, initialise_ray
from rllib_emecom.utils.register_envs import get_registered_env_creator
from rllib_emecom.train.configs import (
    create_args_parser, get_algo_config, Policies, EnvConfig
)

from argparse import Namespace
from typing import Tuple
import os

import ray
from ray import tune


def get_env_config(args: Namespace) -> Tuple[Policies, EnvConfig]:
    env_config = {
        'render_mode': 'rgb_array',
        "world_shape": [args.grid_size, args.grid_size],
        "num_agents": args.n_agents,
        "max_episode_len": args.max_episode_len,
        "goal_shift": args.goal_shift,
        "scalar_obs": args.scalar_obs,
    }
    env_name = args.env
    env = get_registered_env_creator(env_name)(env_config)
    return env_config, [f'agent_{i}' for i in range(args.n_agents)]


def parse_args() -> Namespace:
    parser = create_args_parser()
    parser.add_argument('--env', type=str, default='goal_comms_gridworld')
    parser.add_argument('--stop_timesteps', type=int, default=5_000_000)
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--goal_shift', type=int, default=1)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--max_episode_len', type=int, default=10)
    parser.add_argument('--scalar_obs', action='store_true', default=False)
    parser.add_argument('--message_dim', type=int, default=8)
    parser.add_argument('--comm_channel_fn', type=str, default='straight_through')
    parser.add_argument('--no_param_sharing', action='store_true', default=False)
    parser.add_argument('--no_custom_module', action='store_true', default=False)
    return parser.parse_args()


def main():
    try:
        initialise_ray()

        args = parse_args()
        config = get_algo_config(args, *get_env_config(args))

        tune.run(
            args.algo.upper(),
            name=args.env,
            stop={
                "timesteps_total": args.stop_timesteps,
            },
            checkpoint_freq=10,
            storage_path=f'{os.getcwd()}/ray_results/{args.env}',
            config=config.to_dict(),
            callbacks=[get_wandb_callback()]
        )

        print('Finished training.')

    finally:
        print('Shutting down Ray.')
        ray.shutdown()


if __name__ == "__main__":
    main()
