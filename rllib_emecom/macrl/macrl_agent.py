from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec
from rllib_emecom.macrl.macrl_action_head import MACRLActionHead

from typing import Tuple

from gymnasium.spaces import Discrete, Box
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig
from ray.rllib.core.models.base import SampleBatch, ENCODER_OUT
from ray.rllib.utils.nested_dict import NestedDict

torch, nn = try_import_torch()


class MACRLAgent(nn.Module):
    """
    MACRL action head takes an encoding of the inputs for an agent,
    and the messages for that agent from the other agents, and returns
    inputs to the agent's action distribution.
    """
    framework: str = "torch"

    def __init__(self,
                 observation_space: Box,
                 action_space: Discrete,
                 catalog: PPOCatalog,
                 comms_spec: CommunicationSpec):
        super().__init__()
        self.catalog = catalog
        self.observation_space = observation_space
        self.action_space = action_space
        self.comms_spec = comms_spec
        self.n_agents = comms_spec.n_agents
        self.message_dim = comms_spec.message_dim

        encoder_config = self.get_inputs_encoder_config()
        self.inputs_encoder = encoder_config.build(framework=self.framework)
        actor_encoding_dim = self.inputs_encoder.get_output_specs()[ENCODER_OUT].shape[-1]
        self.msgs_fn = self.build_outgoing_msgs_fn(actor_encoding_dim)
        self.action_head = MACRLActionHead(action_space,
                                           actor_encoding_dim,
                                           self.catalog,
                                           self.comms_spec)

    def build_outgoing_msgs_fn(self, actor_encoding_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(actor_encoding_dim, self.n_agents * self.message_dim)
        )

    def get_inputs_encoder_config(self) -> ModelConfig:
        return self.catalog._get_encoder_config(
            observation_space=self.observation_space,
            model_config_dict=self.catalog._model_config_dict,
            action_space=self.action_space,
            view_requirements=self.catalog._view_requirements
        )

    def encode_inputs_and_create_msgs(
            self, inputs: NestedDict
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_id: agent to run
            inputs: observations and state for the agent

        Returns:
            inps_encoding: tensor of shape `batch_size x actor_encoding_dim`
            out_msgs: tensor of shape `batch_size x n_agents x message_dim`
        """
        device = inputs[SampleBatch.OBS].device
        self.inputs_encoder.to(device)
        self.msgs_fn.to(device)

        inps_encoding = self.inputs_encoder(inputs)[ENCODER_OUT]
        out_msgs = self.msgs_fn(inps_encoding)
        out_msgs = out_msgs.reshape(-1, self.n_agents, self.message_dim)
        return inps_encoding, out_msgs

    def act(self,
            inputs_encoding: torch.Tensor,
            msgs_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs_encoding: agent's inputs encoding
            msgs_in: a tensor with dims `batch_size x n_agents x message_dim`
                that contains messages from each of the agents in the env

        Returns:
            A tensor with dims `batch_size x n_actions` of action logits
        """
        return self.action_head(inputs_encoding, msgs_in)
