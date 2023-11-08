from rllib_emecom.macrl import AgentID
from rllib_emecom.macrl.comms.comms_spec import CommNetwork
from rllib_emecom.macrl.ppo.macrl_ppo import PPOMACRLConfig
from rllib_emecom.utils.video_callback import VideoEvaluationsCallback
from rllib_emecom.utils.experiment_utils import WANDB_PROJECT

from typing import Any, Dict, List
from argparse import ArgumentParser, Namespace

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

EnvConfig = Dict[str, Any]


def add_nn_args(parser: ArgumentParser):
    parser.add_argument('--n_fc_layers', type=int, default=2)
    parser.add_argument('--fc_size', type=int, default=256)
    # parser.add_argument('--not_use_lstm', action='store_true', default=False)
    # parser.add_argument('--lstm_size', type=int, default=64)


def add_ppo_args(parser: ArgumentParser):
    parser.add_argument('--lambda_coeff', type=float, default=0.95)
    parser.add_argument('--not_use_gae', action='store_true', default=False)
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--grad_clip', type=float, default=None)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--vf_loss_coeff', type=float, default=0.25)
    parser.add_argument('--kl_coeff', type=float, default=0.0)
    parser.add_argument('--train_batch_size', type=int, default=10_000)
    parser.add_argument('--sgd_minibatch_size', type=int, default=2048)
    parser.add_argument('--num_sgd_iters', type=int, default=5)


def add_macrl_args(parser: ArgumentParser):
    parser.add_argument('--message_dim', type=int, default=8)
    parser.add_argument('--comm_channel_fn', type=str, default='gumbel_softmax')

    parser.add_argument('--comm_channel_temp', type=float, default=2.0)
    parser.add_argument('--comm_channel_annealing', action='store_true', default=False)
    parser.add_argument('--comm_channel_end_temp', type=float, default=.5)
    parser.add_argument('--comm_channel_annealing_start_iter', type=int, default=0)
    parser.add_argument('--comm_channel_annealing_iters', type=int, default=500)

    parser.add_argument('--comm_channel_noise', type=float, default=0.25)
    parser.add_argument('--comm_channel_activation', type=str, default='tanh')
    parser.add_argument('--no_param_sharing', action='store_true', default=False)


def create_default_args_parser() -> ArgumentParser:
    algo_parser = ArgumentParser()
    algo_parser.add_argument('--algo', type=str, default='ppo',
                             choices=['ppo', 'maddpg', 'dqn'],
                             help='RL Algorithm to use (default: %(default)s)')
    algo = algo_parser.parse_known_args()[0].algo

    parser = ArgumentParser()
    parser.add_argument('--algo', type=str, default='ppomacrl',
                        choices=['ppomacrl', 'maddpg', 'dqn'],
                        help='RL Algorithm to use (default: %(default)s)')

    parser.add_argument('--learning_rate', type=float, default=5e-4)
    parser.add_argument('--gamma', type=float, default=0.99)

    parser.add_argument('--num_rollout_workers', type=int, default=8)
    parser.add_argument('--evaluation_interval', type=int, default=10)
    parser.add_argument('--evaluation_duration', type=int, default=25)
    parser.add_argument('--evaluation_num_workers', type=int, default=4)
    parser.add_argument('--stop_timesteps', type=int, default=5_000_000)

    parser.add_argument('--no_wandb', action='store_true', default=False)
    parser.add_argument('--wandb_project', type=str, default=WANDB_PROJECT)

    if algo == 'ppo':
        add_ppo_args(parser)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')

    add_nn_args(parser)
    add_macrl_args(parser)

    return parser


def get_ppo_macrl_config(args: Namespace,
                         env_config: EnvConfig,
                         agent_ids: List[AgentID],
                         comm_channels: CommNetwork = None) -> PPOMACRLConfig:

    model_config = {
        'fcnet_hiddens': [args.fc_size] * args.n_fc_layers,
        'vf_share_layers': False,
    }

    return (
        PPOMACRLConfig()
        .environment(
            args.env,
            env_config=env_config,
            disable_env_checking=True,
            clip_actions=True
        )
        .training(
            train_batch_size=args.train_batch_size,
            lr=args.learning_rate,
            gamma=args.gamma,
            lambda_=args.lambda_coeff,
            use_gae=not args.not_use_gae,
            clip_param=args.clip_param,
            grad_clip=args.grad_clip,
            entropy_coeff=args.entropy_coeff,
            vf_loss_coeff=args.vf_loss_coeff,
            sgd_minibatch_size=args.sgd_minibatch_size,
            kl_coeff=args.kl_coeff,
            num_sgd_iter=args.num_sgd_iters,
            model=model_config,
            _enable_learner_api=True,
        )
        .rollouts(
            num_rollout_workers=args.num_rollout_workers,
            rollout_fragment_length='auto',
        )
        .communications(
            agent_ids=agent_ids,
            comm_channels=comm_channels,
            message_dim=args.message_dim,
            channel_fn=args.comm_channel_fn,
            channel_fn_config={
                'temperature': args.comm_channel_temp,
                'temperature_annealing': args.comm_channel_annealing,
                'annealing_start_iter': args.comm_channel_annealing_start_iter,
                'n_anneal_iterations': args.comm_channel_annealing_iters,
                'final_temperature': args.comm_channel_end_temp,
                'channel_noise': args.comm_channel_noise,
                'channel_activation': args.comm_channel_activation
            },
        )
        .resources(
            num_gpus=1,
            num_gpus_per_learner_worker=1
        )
        .callbacks(VideoEvaluationsCallback)
        .evaluation(
            evaluation_interval=args.evaluation_interval,
            evaluation_duration=args.evaluation_duration,
            evaluation_num_workers=args.evaluation_num_workers,
        )
    )


def get_algo_config(args: Namespace,
                    env_config: dict,
                    agent_ids: List[AgentID]) -> AlgorithmConfig:
    algo = args.algo.lower()
    if algo == 'ppomacrl':
        return get_ppo_macrl_config(args, env_config, agent_ids)
    else:
        raise ValueError(f'Unknown algorithm: {algo}')
