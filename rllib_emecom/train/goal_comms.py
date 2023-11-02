from rllib_emecom.macrl import AgentID
from rllib_emecom.utils.experiment_utils import run_training, initialise_ray
from rllib_emecom.utils.comms_renderer import CommsRenderer
from rllib_emecom.train.configs import (
    create_default_args_parser, get_algo_config, EnvConfig
)

from argparse import ArgumentParser, Namespace
from typing import List, Optional, Tuple

from ray.rllib.evaluate import rollout
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig


def get_env_config(args: Namespace) -> Tuple[EnvConfig, List[AgentID]]:
    env_config = {
        'render_mode': 'rgb_array',
        'world_shape': [args.grid_size, args.grid_size],
        'num_agents': args.n_agents,
        'max_episode_len': args.max_episode_len,
        'goal_shift': args.goal_shift,
        'scalar_obs': args.scalar_obs,
        'observe_others_pos': not args.only_obs_self_pos,
        'observe_goals': args.observe_goals,
        'render_config': {
            'renderer_cls': CommsRenderer,
            'n_msgs': args.message_dim,
            'fps': 2,
            'episodes_per_video': 6
        }
    }
    return env_config, [f'agent_{i}' for i in range(args.n_agents)]


def create_goal_comms_args_parser() -> ArgumentParser:
    parser = create_default_args_parser()
    parser.add_argument('--env', type=str, default='goal_comms_gridworld')
    parser.add_argument('--n_agents', type=int, default=3)
    parser.add_argument('--goal_shift', type=int, default=1)
    parser.add_argument('--grid_size', type=int, default=5)
    parser.add_argument('--max_episode_len', type=int, default=10)
    parser.add_argument('--scalar_obs', action='store_true', default=False)
    parser.add_argument('--only_obs_self_pos', action='store_true', default=False)
    parser.add_argument('--observe_goals', action='store_true', default=False)
    return parser


def parse_args() -> Namespace:
    parser = create_goal_comms_args_parser()
    args, _ = parser.parse_known_args()
    return args


def get_goal_comms_config(args: Optional[Namespace] = None) -> AlgorithmConfig:
    args = args or parse_args()
    return get_algo_config(args, *get_env_config(args))


def train_goal_comms():
    args = parse_args()
    config = get_goal_comms_config(args)
    run_training(config, args)


def run_rollout_test(test_args: Namespace):
    args = parse_args()
    args.num_rollout_workers = 0
    args.evaluation_num_workers = 0
    config = get_goal_comms_config(args)
    initialise_ray()
    algo = config.build()
    rollout(algo, args.env,
            num_episodes=test_args.n_episodes,
            num_steps=test_args.n_steps)


def run_train_test(test_args: Namespace):
    args = parse_args()
    args.num_rollout_workers = 0
    args.evaluation_num_workers = 0
    args.train_batch_size = 1000
    args.sgd_minibatch_size = 32
    config = get_goal_comms_config(args)
    initialise_ray()
    algo = config.build()
    algo.train()


def get_test_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--train_test', action='store_true', default=False)
    parser.add_argument('--rollout_test', action='store_true', default=False)
    parser.add_argument('--n_episodes', type=int, default=10)
    parser.add_argument('--n_steps', type=int, default=1000)
    test_args, _ = parser.parse_known_args()
    return test_args


if __name__ == "__main__":
    test_args = get_test_args()
    if test_args.rollout_test:
        run_rollout_test(test_args)
    elif test_args.train_test:
        run_train_test(test_args)
    else:
        train_goal_comms()
