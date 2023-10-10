from rllib_emecom.utils.experiment_utils import get_wandb_callback, initialise_ray
from rllib_emecom.utils.comms_renderer import CommsRenderer
from rllib_emecom.train.configs import (
    create_args_parser, get_algo_config, Policies, EnvConfig
)

from argparse import ArgumentParser, Namespace
from typing import Tuple
import os

import ray
from ray import tune
from ray.rllib.evaluate import rollout


def get_env_config(args: Namespace) -> Tuple[Policies, EnvConfig]:
    env_config = {
        'render_mode': 'rgb_array',
        'world_shape': [args.grid_size, args.grid_size],
        'num_agents': args.n_agents,
        'max_episode_len': args.max_episode_len,
        'goal_shift': args.goal_shift,
        'scalar_obs': args.scalar_obs,
        'render_config': {
            'renderer_cls': CommsRenderer,
            'n_msgs': args.message_dim,
            'fps': 2,
            'episodes_per_video': 6
        }
    }
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
    args, _ = parser.parse_known_args()
    return args


def run_experiment():
    try:
        initialise_ray()

        args = parse_args()
        config = get_algo_config(args, *get_env_config(args))

        tune.run(
            args.algo.upper(),
            name=args.env,
            stop={'timesteps_total': args.stop_timesteps},
            checkpoint_freq=10,
            storage_path=f'{os.getcwd()}/ray_results/{args.env}',
            config=config.to_dict(),
            callbacks=[get_wandb_callback()]
        )

        print('Finished training.')

    finally:
        print('Shutting down Ray.')
        ray.shutdown()


def run_rollout_test(test_args: Namespace):
    initialise_ray()
    args = parse_args()
    config = get_algo_config(args, *get_env_config(args))
    algo = config.build()
    rollout(algo, args.env,
            num_episodes=test_args.n_episodes,
            num_steps=test_args.n_steps)


def get_test_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--rollout_test', action='store_true', default=False)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=1000)
    test_args, _ = parser.parse_known_args()
    return test_args


if __name__ == "__main__":
    test_args = get_test_args()
    if test_args.rollout_test:
        run_rollout_test(test_args)
    else:
        run_experiment()
