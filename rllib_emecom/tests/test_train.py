from .utils import ray_init_wrapper, parse_default_args
from rllib_emecom.macrl.comms.comms_spec import CommNetwork
from rllib_emecom.macrl.macrl_agent import MACRLAgent
from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule
from rllib_emecom.macrl.ppo.macrl_ppo import PPOMACRLConfig
from rllib_emecom.train.goal_comms import (
    create_goal_comms_args_parser, get_goal_comms_config, get_env_config
)
from rllib_emecom.train.configs import get_ppo_macrl_config
from rllib_emecom.macrl.comms import (
    get_channel_fn_cls,
    GumbelSoftmaxCommunicationChannel,
    DiscretiseRegulariseCommunicationChannel
)

import random
from ray.rllib.evaluation.rollout_worker import RolloutWorker


def get_macrl_ppo_module(algo):

    assert isinstance(algo.config, PPOMACRLConfig)

    local_worker = algo.workers.local_worker()
    assert isinstance(local_worker, RolloutWorker), \
        'Expected local worker to be a RolloutWorker'

    # all agents have to be a single logical policy for communication
    assert set(local_worker.policy_dict.keys()) == {'default_policy'}, \
        'Expected only one policy to be created'

    policy = local_worker.get_policy('default_policy')

    assert isinstance(policy.model, PPOTorchMACRLModule), \
        'Expected policy model to be a PPOTorchMACRLModule'

    return policy.model


@ray_init_wrapper
def test_create_ppo_macrl_config():
    parser = create_goal_comms_args_parser()
    args = parse_default_args(parser,
                              train_batch_size=256,
                              sgd_minibatch_size=32,
                              num_rollout_workers=0,
                              n_agents=3,
                              evaluation_num_workers=0)
    config = get_ppo_macrl_config(args, *get_env_config(args))
    algo = config.build()
    ppo_macrl_module = get_macrl_ppo_module(algo)

    assert ppo_macrl_module.agents.keys() == {f'agent_{i}' for i in range(args.n_agents)}, \
        'Expected policy model to have one agent per agent ID'

    assert all(isinstance(agent, MACRLAgent) for agent in ppo_macrl_module.agents.values()), \
        'Expected policy model to have MACRLAgents'


@ray_init_wrapper
def test_create_comms_args_macrl_config():

    def get_random_comm_network(n_agents: int) -> CommNetwork:
        agent_ids = [f'agent_{i}' for i in range(n_agents)]
        return {
            agent_id: [
                other_id for other_id in agent_ids
                if other_id != agent_id and random.random() > 0.5
            ]
            for agent_id in agent_ids
        }

    comm_specs = [
        {
            'n_agents': 2,
            'message_dim': 5,
            'comm_channel_fn': 'straight_through',
            'channel_fn_config': {},
            'comm_channels': get_random_comm_network(2)
        },
        {
            'n_agents': 5,
            'message_dim': 10,
            'comm_channel_fn': 'gumbel_softmax',
            'comm_channel_temp': 2.5,
            'comm_channel_end_temp': 0.3,
            'comm_channel_no_annealing': True,
            'comm_channel_annealing_start_iter': 1000,
            'comm_channel_annealing_iters': 2000,
            'comm_channels': get_random_comm_network(5)
        },
        {
            'n_agents': 3,
            'message_dim': 16,
            'comm_channel_fn': 'dru',
            'comm_channel_noise': 1.5,
            'comm_channel_activation': 'sigmoid',
            'comm_channels': get_random_comm_network(3)
        }
    ]

    for comm_spec_args in comm_specs:
        comm_channels = comm_spec_args.pop('comm_channels')
        parser = create_goal_comms_args_parser()
        args = parse_default_args(parser,
                                  train_batch_size=256,
                                  sgd_minibatch_size=32,
                                  num_rollout_workers=0,
                                  evaluation_num_workers=0,
                                  **comm_spec_args)
        config = get_ppo_macrl_config(args, *get_env_config(args),
                                      comm_channels=comm_channels)
        algo = config.build()
        ppo_macrl_module = get_macrl_ppo_module(algo)
        assert hasattr(ppo_macrl_module, 'comm_channel_fn'), \
            'Expected module to have a channel_fn attribute'

        comm_channel_fn = ppo_macrl_module.comm_channel_fn
        expected_comm_channel_fn_cls = get_channel_fn_cls(comm_spec_args['comm_channel_fn'])
        assert isinstance(comm_channel_fn, expected_comm_channel_fn_cls), \
            f'Expected module to have a channel_fn attribute of type {expected_comm_channel_fn_cls} ' \
            f'but got {type(comm_channel_fn)}'

        if expected_comm_channel_fn_cls == GumbelSoftmaxCommunicationChannel:
            does_gumbel_softmax_channel_match_config(comm_channel_fn, comm_spec_args)

        if expected_comm_channel_fn_cls == DiscretiseRegulariseCommunicationChannel:
            does_dru_channel_match_config(comm_channel_fn, comm_spec_args)


def does_gumbel_softmax_channel_match_config(channel: GumbelSoftmaxCommunicationChannel,
                                             config: dict):
    annealing = not config['comm_channel_no_annealing']
    if annealing:
        assert channel.start_temperature == config['comm_channel_temp']
        assert channel.final_temperature == config['comm_channel_end_temp']
        assert channel.annealing_start_iter == config['comm_channel_annealing_start_iter']
        assert channel.n_anneal_iterations == config['comm_channel_annealing_iters']

    else:
        assert channel.temperature == config['comm_channel_temp']


def does_dru_channel_match_config(channel: DiscretiseRegulariseCommunicationChannel,
                                  config: dict):
    assert channel.channel_noise == config['comm_channel_noise']
    assert channel.channel_activation == config['comm_channel_activation']


@ray_init_wrapper
def test_train_step():
    parser = create_goal_comms_args_parser()
    args = parse_default_args(parser,
                              train_batch_size=256,
                              sgd_minibatch_size=32,
                              num_rollout_workers=0,
                              evaluation_num_workers=0)
    config = get_goal_comms_config(args)
    algo = config.build()
    algo.train()
