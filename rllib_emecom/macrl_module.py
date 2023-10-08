from rllib_emecom.comm_network import CommunicationSpec

from typing import Any, Dict, List, Mapping

from gymnasium.spaces import Box, Discrete
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.core.rl_module.marl_module import ModuleID
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.models.base import (
    SampleBatch, ENCODER_OUT
)
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog, _check_if_diag_gaussian
from ray.rllib.core.models.configs import MLPHeadConfig, FreeLogStdMLPHeadConfig


torch, nn = try_import_torch()

TRAIN_FORWARD = "forward_train"
INFERENCE_FORWARD = "forward_inference"
EXPLORATION_FORWARD = "forward_exploration"

AgentID = str


class PPOTorchMACRLModule(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def get_comms_spec(self) -> CommunicationSpec:
        assert hasattr(self.config, 'model_config_dict'), \
            "Expected model_config_dict to be set on module config."
        assert 'communication_spec' in self.config.model_config_dict, \
            "Expected communication_spec to be set on model_config_dict."

        comm_spec = self.config.model_config_dict['communication_spec']

        assert isinstance(comm_spec, CommunicationSpec), \
            "Expected communication_spec to be a CommunicationSpec."

        return comm_spec

    def get_n_agents(self) -> int:
        return self.get_comms_spec().n_agents

    def get_message_dim(self) -> int:
        return self.get_comms_spec().message_dim

    def get_comm_network(self) -> Dict[ModuleID, List[ModuleID]]:
        return self.get_comms_spec().comm_channels

    def get_agent_ids(self) -> List[AgentID]:
        return self.get_comms_spec().agents

    def _build_pi_head(self, catalog: PPOCatalog, action_space):
        # Get action_distribution_cls to find out about the output dimension for pi_head
        action_distribution_cls = catalog.get_action_dist_cls(framework=self.framework)
        if catalog._model_config_dict["free_log_std"]:
            _check_if_diag_gaussian(
                action_distribution_cls=action_distribution_cls, framework=self.framework
            )
        assert isinstance(action_space, Discrete), \
            "Expected action space to be a Discrete."
        required_output_dim = int(action_space.n)
        # Now that we have the action dist class and number of outputs, we can define
        # our pi-config and build the pi head.
        pi_head_config_class = (
            FreeLogStdMLPHeadConfig
            if catalog._model_config_dict["free_log_std"]
            else MLPHeadConfig
        )
        pi_and_vf_head_hiddens = catalog._model_config_dict["post_fcnet_hiddens"]
        pi_and_vf_head_activation = catalog._model_config_dict["post_fcnet_activation"]
        pi_head_config = pi_head_config_class(
            input_dims=catalog.latent_dims,
            hidden_layer_dims=pi_and_vf_head_hiddens,
            hidden_layer_activation=pi_and_vf_head_activation,
            output_layer_dim=required_output_dim,
            output_layer_activation="linear",
        )

        return pi_head_config.build(framework=self.framework)

    @override(PPORLModule)
    def setup(self):
        catalog: PPOCatalog = self.config.get_catalog()
        assert isinstance(catalog, PPOCatalog), \
            "Expected catalog to be a PPOCatalog."

        self.n_agents = self.get_n_agents()
        self.message_dim = self.get_message_dim()
        self.comm_network = self.get_comms_spec().comm_channels

        all_obs_shape = catalog.observation_space.shape
        assert isinstance(catalog.observation_space, Box), \
            "Expected observation space to be a Box."
        ind_obs_space = Box(
            catalog.observation_space.low.min(),
            catalog.observation_space.high.max(),
            shape=(*all_obs_shape[:-1], all_obs_shape[0] // self.n_agents,)
        )
        ind_act_space = catalog.action_space[0]
        actor_config = catalog._get_encoder_config(
            observation_space=ind_obs_space,
            model_config_dict=catalog._model_config_dict,
            action_space=ind_act_space,
            view_requirements=catalog._view_requirements
        )

        self.actor_encoder = actor_config.build(framework=self.framework)
        self.pi_head = self._build_pi_head(catalog, action_space=ind_act_space)

        self.critic_encoder = catalog._encoder_config.build(framework=self.framework)
        self.vf_head = catalog.build_vf_head(framework=self.framework)

        self.action_dist_cls = catalog.get_action_dist_cls(framework=self.framework)
        # assumes that actor and critic encoders are not shared
        actor_encoding_dim = self.actor_encoder.get_output_specs()[ENCODER_OUT].shape[-1]
        # critic_encoding_dim = self.critic_encoder.get_output_specs()[ENCODER_OUT].shape[-1]

        self.create_outgoing_msgs = nn.Sequential(
            nn.Linear(actor_encoding_dim, self.n_agents * self.message_dim)
        )
        hidden_dim = 256
        self.aggregate_msgs_and_obs = nn.Sequential(
            nn.Linear(self.n_agents * self.message_dim + actor_encoding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, actor_encoding_dim),
            nn.ReLU(),
        )

    @override(RLModule)
    def get_initial_state(self) -> dict:
        return {}

    def encode_obs_and_produce_msgs(self, inputs):
        obs_encoding = self.actor_encoder(inputs)[ENCODER_OUT]
        out_msgs = self.create_outgoing_msgs(obs_encoding)
        out_msgs = out_msgs.reshape(-1, self.n_agents, self.message_dim)
        return obs_encoding, out_msgs

    def pi(self, obs_encoding, msgs_in):
        agent_msgs_in = msgs_in.view(-1, self.n_agents * self.message_dim)
        processor_inp = torch.cat([agent_msgs_in, obs_encoding], dim=-1)
        final_encoding = self.aggregate_msgs_and_obs(processor_inp)
        return self.pi_head(final_encoding)

    def vf(self, all_agents_obs_enc):
        return self.vf_head(all_agents_obs_enc).squeeze(-1)

    def get_comm_mask(self, agent_id: AgentID, batch_size: int = 1) -> torch.Tensor:
        mask = torch.zeros(batch_size, self.n_agents, self.message_dim)
        for i in range(self.n_agents):
            if self.get_comms_spec().index_to_agent(i) in self.comm_network[agent_id]:
                mask[:, i] = torch.ones(self.message_dim)
        return mask

    def _handle_message_passing(self, msgs_out: NestedDict):
        """
        """
        msgs_in = {}

        agent_ids = list(msgs_out.keys())

        # mask outgoing messages that are not permitted by the comm spec
        for agent_id in agent_ids:
            batch_size, *_ = msgs_out[agent_id].shape
            mask = self.get_comm_mask(agent_id, batch_size)
            msgs_out[agent_id] = mask * msgs_out[agent_id]

        for agent_id in agent_ids:
            msgs_in[agent_id] = {}

            # collect messages from other agents
            i = self.get_comms_spec().get_agent_idx(agent_id)
            msgs_in[agent_id] = torch.cat([
                msgs_out[other_id][:, i:i + 1]
                for other_id in agent_ids
            ], dim=-2)

        # TODO: add channel function

        return msgs_in

    def _actors_forward(self, batch: NestedDict[Any]):
        msgs_out = {}
        encodings = {}
        for agent_id in self.get_agent_ids():
            obs_enc, msgs = self.encode_obs_and_produce_msgs(batch[agent_id])
            encodings[agent_id] = obs_enc
            msgs_out[agent_id] = msgs

        msgs_in = self._handle_message_passing(msgs_out)

        outputs = {}
        for agent_id, obs_enc in encodings.items():
            outputs[agent_id] = {}
            outputs[agent_id][SampleBatch.ACTION_DIST_INPUTS] = \
                self.pi(obs_enc, msgs_in[agent_id])

        return outputs

    def _critics_forward(self, all_agents_obs):
        all_agents_obs_enc = self.critic_encoder(all_agents_obs)[ENCODER_OUT]
        return self.vf_head(all_agents_obs_enc).squeeze(-1)

    def nest_by_agent(self, batch):
        obs = batch[SampleBatch.OBS]
        inputs = {}
        for i, agent_id in enumerate(self.get_agent_ids()):
            inputs[agent_id] = {}
            if len(obs.shape) == 2:
                obs_dim = obs.shape[-1] // self.n_agents
                inputs[agent_id][SampleBatch.OBS] = obs[:, i * obs_dim:(i + 1) * obs_dim]
            else:
                inputs[agent_id][SampleBatch.OBS] = obs[:, i]
        return inputs

    @override(RLModule)
    def _forward_inference(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch)

    def _common_forward(self, batch: NestedDict[Any]) -> Mapping[str, Any]:
        """
        This forward method
        """
        outputs = {}
        actor_inputs = self.nest_by_agent(batch)
        actor_outputs = self._actors_forward(actor_inputs)
        outputs[SampleBatch.VF_PREDS] = self._critics_forward(batch)
        outputs[SampleBatch.ACTION_DIST_INPUTS] = torch.cat([
            outs[SampleBatch.ACTION_DIST_INPUTS] for outs in actor_outputs.values()
        ], axis=-1)
        return outputs
