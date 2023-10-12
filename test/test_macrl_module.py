from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule, COMMS_STATE
from rllib_emecom.macrl.comm_network import CommunicationSpec

import gymnasium as gym
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
    agent_ids = [f'agent_{i}' for i in range(n_agents)]

    comm_channels = {
        agent_id: [
            other_id for other_id in agent_ids
            if other_id != agent_id
        ]
        for agent_id in agent_ids
    }

    comm_spec = CommunicationSpec(
        message_dim=message_dim,
        comm_channels=comm_channels,
        static=True,
        channel_fn=channel_fn,
        channel_fn_config=channel_config
    )

    model_config_dict = {"fcnet_hiddens": [32] * 2}

    env = gym.make("CartPole-v1")

    module_specs = {
        agent_id: SingleAgentRLModuleSpec(
            module_class=PPOTorchRLModule,
            catalog_class=PPOCatalog,
            observation_space=env.observation_space,
            action_space=env.action_space,
            model_config_dict={
                'communication_spec': comm_spec,
                **model_config_dict
            }
        )
        for agent_id in agent_ids
    }

    spec = MultiAgentRLModuleSpec(
        marl_module_class=PPOTorchMACRLModule,
        module_specs=module_specs,
    )

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
