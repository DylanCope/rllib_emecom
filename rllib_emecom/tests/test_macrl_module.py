from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec
from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule
from rllib_emecom.macrl.macrl_module_spec import MACRLModuleSpec
from rllib_emecom.macrl.macrl_config import get_fully_connected_comm_channels

from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog

from gymnasium.spaces import Tuple, Box, Discrete
import torch


def create_test_macrl_module(
        n_agents=3, message_dim=5,
        channel_fn='straight_through',
        **channel_config) -> PPOTorchMACRLModule:
    agent_ids = [f'agent_{i}' for i in range(n_agents)]
    mock_obs_space = Box(low=0, high=1, shape=(8 * n_agents,))
    mock_act_space = Tuple([Discrete(2) for _ in range(n_agents)])

    comm_spec = CommunicationSpec(
        message_dim=message_dim,
        comm_channels=get_fully_connected_comm_channels(agent_ids),
        channel_fn=channel_fn,
        channel_fn_config=channel_config
    )

    model_config_dict = {}

    spec = MACRLModuleSpec(comm_spec=comm_spec,
                           observation_space=mock_obs_space,
                           action_space=mock_act_space,
                           model_config_dict=model_config_dict,
                           module_class=PPOTorchMACRLModule,
                           catalog_class=PPOCatalog)

    return spec.build(), agent_ids


def test_handle_message_passing():
    batch_size = 2
    n_agents = 3
    message_dim = 5
    module, agent_ids = create_test_macrl_module(n_agents, message_dim)

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


def test_handle_message_passing_with_time_dim():
    batch_size = 2
    n_agents = 3
    message_dim = 5
    time_steps = 4
    module, agent_ids = create_test_macrl_module(n_agents, message_dim)

    msgs_out = {
        agent_id: torch.randn(batch_size, time_steps, n_agents, message_dim)
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
            outgoing_msg = msgs_out[agent_1][:, :, j:j + 1]
            incoming_msg = msgs_in[agent_2][:, :, i:i + 1]
            assert torch.allclose(outgoing_msg, incoming_msg)
