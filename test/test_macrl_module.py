from .utils import parse_default_args
from rllib_emecom.train.configs import get_ppo_macrl_module_spec, create_default_args_parser

from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule
from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec

from gymnasium.spaces import Tuple, Box, Discrete
import torch
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


def create_mock_macrl_module(
        n_agents=3, message_dim=5,
        channel_fn='straight_through',
        **channel_config
        ) -> PPOTorchMACRLModule:
    parser = create_default_args_parser()
    args = parse_default_args(parser, **{
        'n_agents': n_agents,
        'message_dim': message_dim,
        'comm_channel_fn': channel_fn,
        **channel_config
    })
    agent_ids = [f'agent_{i}' for i in range(n_agents)]
    mock_obs_space = Box(low=0, high=1, shape=(8*n_agents,))
    mock_act_space = Tuple([Discrete(2) for _ in range(n_agents)])
    spec = get_ppo_macrl_module_spec(args, agent_ids,
                                     observation_space=mock_obs_space,
                                     action_space=mock_act_space)
    return spec.build(), agent_ids


def test_handle_message_passing():
    batch_size = 2
    n_agents = 3
    message_dim = 5
    module, agent_ids = create_mock_macrl_module(n_agents, message_dim)

    msgs_out = {
        agent_id: torch.randn(batch_size, n_agents, message_dim)
        for agent_id in agent_ids
    }

    # Call the _handle_message_passing method
    msgs_in = module._handle_message_passing(msgs_out)

    # Check that the output is a dictionary with the same keys as msgs_in
    assert isinstance(msgs_in, dict)
    assert set(msgs_in.keys()) == set(msgs_out.keys())

    comm_spec = module.get_comms_spec()

    # Check that the comms_state values are correct
    for agent_1 in msgs_out.keys():
        i = comm_spec.get_agent_idx(agent_1)
        for agent_2 in msgs_in.keys():
            j = comm_spec.get_agent_idx(agent_2)
            outgoing_msg = msgs_out[agent_1][:, j:j + 1]
            incoming_msg = msgs_in[agent_2][:, i:i + 1]
            assert torch.allclose(outgoing_msg, incoming_msg)
