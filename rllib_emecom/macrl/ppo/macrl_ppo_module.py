from rllib_emecom.macrl import AgentID
from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec
from rllib_emecom.macrl.macrl_action_head import MACRLActionHead

from typing import Any, Dict, List, Mapping, Optional, Union, Tuple as TupleType
from gymnasium.spaces import Box, Discrete, Tuple

from ray.rllib.core.rl_module.rl_module import RLModule
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.core.models.base import SampleBatch, ENCODER_OUT
from ray.rllib.core.rl_module.torch import TorchRLModule
from ray.rllib.algorithms.ppo.ppo_rl_module import PPORLModule
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.core.models.configs import ModelConfig

import numpy as np


torch, nn = try_import_torch()

TRAIN_FORWARD = "forward_train"
INFERENCE_FORWARD = "forward_inference"
EXPLORATION_FORWARD = "forward_exploration"

ACTOR_ENCODER = 'actor_encoder'
MSGS_FN = 'msgs_creator'
MSG_OBS_AGG = 'msg_and_obs_aggregator'
ACTION_HEAD = 'pi_head'

MSGS_SENT = "comms_state"

DEFAULT_CONFIG = {
    'share_actor_params': True
}


class PPOTorchMACRLModule(TorchRLModule, PPORLModule):
    framework: str = "torch"

    def get_model_config_dict(self) -> dict:
        if not hasattr(self, 'catalog'):
            self.catalog: PPOCatalog = self.config.get_catalog()

        assert hasattr(self.catalog, '_model_config_dict'), \
            "Expected model_config_dict to be set on catalog."
        return {**DEFAULT_CONFIG, **self.catalog._model_config_dict}

    def get_comms_spec(self) -> CommunicationSpec:
        model_config_dict = self.get_model_config_dict()

        assert 'communication_spec' in model_config_dict, \
            "Expected communication_spec to be set on model_config_dict."

        comm_spec = model_config_dict['communication_spec']

        assert isinstance(comm_spec, CommunicationSpec), \
            "Expected communication_spec to be a CommunicationSpec."

        return comm_spec

    def get_agent_ids(self) -> List[AgentID]:
        return self.comms_spec.agents

    def _get_individual_obs_space(self) -> Box:
        all_obs_shape = self.catalog.observation_space.shape
        assert isinstance(self.catalog.observation_space, Box), \
            "Expected observation space to be a Box."

        return Box(
            self.catalog.observation_space.low.min(),
            self.catalog.observation_space.high.max(),
            shape=(*all_obs_shape[:-1], all_obs_shape[0] // self.n_agents,)
        )

    def _get_individual_act_space(self) -> Discrete:
        assert isinstance(self.catalog.action_space, Tuple), \
            "Expected action space to be a Discrete."
        return self.catalog.action_space[0]

    def _get_actor_encoder_config(self) -> ModelConfig:
        ind_obs_space = self._get_individual_obs_space()
        ind_act_space = self._get_individual_act_space()

        return self.catalog._get_encoder_config(
            observation_space=ind_obs_space,
            model_config_dict=self.catalog._model_config_dict,
            action_space=ind_act_space,
            view_requirements=self.catalog._view_requirements
        )

    def build_outgoing_msgs_fn(self, actor_encoding_dim: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(actor_encoding_dim, self.n_agents * self.message_dim)
        )

    def build_agent_models(self) -> Dict[AgentID, Dict[str, nn.Module]]:
        """
        Creates models for each agent for computing actor forwards.
        Handles parameter sharing configuration between agents.

        Returns:
            A dictionary where keys are agent ids and values are dictionaries
            from model ids to torch models.
        """
        actor_config = self._get_actor_encoder_config()
        self.share_actor_params = self.get_model_config_dict()['share_actor_params']
        ind_action_space = self._get_individual_act_space()
        agent_models = {}
        if self.share_actor_params:
            actor_encoder = actor_config.build(framework=self.framework)
            actor_encoding_dim = actor_encoder.get_output_specs()[ENCODER_OUT].shape[-1]
            msgs_fn = self.build_outgoing_msgs_fn(actor_encoding_dim)
            action_head = MACRLActionHead(ind_action_space,
                                          actor_encoding_dim,
                                          self.catalog,
                                          self.comms_spec)
            for agent_id in self.comms_spec.agents:
                agent_models[agent_id] = {
                    ACTOR_ENCODER: actor_encoder,
                    MSGS_FN: msgs_fn,
                    ACTION_HEAD: action_head,
                }

        else:
            for agent_id in self.comms_spec.agents:
                actor_encoder = actor_config.build(framework=self.framework)
                actor_encoding_dim = actor_encoder.get_output_specs()[ENCODER_OUT].shape[-1]
                agent_models[agent_id] = {
                    ACTOR_ENCODER: actor_encoder,
                    MSGS_FN: self.build_outgoing_msgs_fn(actor_encoding_dim),
                    ACTION_HEAD: MACRLActionHead(ind_action_space,
                                                 actor_encoding_dim,
                                                 self.catalog,
                                                 self.comms_spec),
                }

        return agent_models

    def set_action_head(self, agent_id: AgentID, action_head: MACRLActionHead):
        self.agent_models[agent_id][ACTION_HEAD] = action_head

    def set_action_encoder(self, agent_id: AgentID, encoder: nn.Module):
        self.agent_models[agent_id][ACTOR_ENCODER] = encoder

    def set_msgs_gn(self, agent_id: AgentID, msgs_fn: nn.Module):
        self.agent_models[agent_id][MSGS_FN] = msgs_fn

    @override(PPORLModule)
    def setup(self):
        self.catalog: PPOCatalog = self.config.get_catalog()
        print('Building MACRL module...')
        self.comms_spec = self.get_comms_spec()
        print(f'Comms spec: {self.comms_spec}')
        self.n_agents = self.comms_spec.n_agents
        self.message_dim = self.comms_spec.message_dim
        self.comm_network = self.comms_spec.comm_channels
        self.comm_channel_fn = self.comms_spec.get_channel_fn()
        print(f'Communication channel function: {self.comm_channel_fn}')

        assert isinstance(self.catalog, PPOCatalog), \
            "Expected catalog to be a PPOCatalog."

        self.critic_encoder = self.catalog._encoder_config.build(framework=self.framework)
        self.vf_head = self.catalog.build_vf_head(framework=self.framework)
        self.action_dist_cls = self.catalog.get_action_dist_cls(framework=self.framework)
        self.agent_models = self.build_agent_models()

        self.last_msgs_sent = None
        self.last_actor_outputs = None
        self.last_inputs = None

    @override(RLModule)
    def get_initial_state(self) -> dict:
        return {}

    def encode_obs_and_produce_msgs(self,
                                    agent_id: AgentID,
                                    inputs: NestedDict) -> TupleType[torch.Tensor, torch.Tensor]:
        """
        Args:
            agent_id: agent to run
            inputs: observations and state for the agent

        Returns:
            obs_encoding: tensor of shape `batch_size x actor_encoding_dim`
            out_msgs: tensor of shape `batch_size x n_agents x message_dim`
        """
        device = inputs[SampleBatch.OBS].device
        self.agent_models[agent_id][ACTOR_ENCODER].to(device)
        self.agent_models[agent_id][MSGS_FN].to(device)

        obs_encoding = self.agent_models[agent_id][ACTOR_ENCODER](inputs)[ENCODER_OUT]
        out_msgs = self.agent_models[agent_id][MSGS_FN](obs_encoding)
        out_msgs = out_msgs.reshape(-1, self.n_agents, self.message_dim)
        return obs_encoding, out_msgs

    def get_action_dist_inputs(self,
                               agent_id: AgentID,
                               obs_encoding: torch.Tensor,
                               msgs_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            agent_id: the agent to compute action distribution inputs for
            obs_encoding: agent's observation encoding
            msgs_in: a tensor with dims `batch_size x n_agents x message_dim`
                that contains messages from each of the agents in the env

        Returns:
            A tensor with dims `batch_size x n_actions` of action logits
        """
        agent_msgs_in = msgs_in.view(-1, self.n_agents * self.message_dim)
        processor_inp = torch.cat([agent_msgs_in, obs_encoding], dim=-1)
        final_encoding = self.agent_models[agent_id][MSG_OBS_AGG](processor_inp)
        return self.agent_models[agent_id][ACTION_HEAD](final_encoding)

    def vf(self, all_agents_obs_enc):
        return self.vf_head(all_agents_obs_enc).squeeze(-1)

    def get_comm_mask(self,
                      agent_id: AgentID,
                      batch_size: int = 1) -> torch.Tensor:
        """
        Creates a mask for removing messages that are not permitted by
        the communication spec.

        Args:
            agent_id: the agent that is sending a message
            batch_size: batch dimension of the output

        Returns:
            A tensor of `batch_size x n_agents x message_dim` in which
            each entry is 1.0 if the sender is permitted to send a
            message to the corresponding receiver, and 0.0 otherwise.
        """

        mask = torch.zeros(batch_size, self.n_agents, self.message_dim)
        for i in range(self.n_agents):
            if self.comms_spec.index_to_agent(i) in self.comm_network[agent_id]:
                mask[:, i] = torch.ones(self.message_dim)
        return mask

    def _handle_message_passing(self,
                                msgs_out: Dict[AgentID, torch.Tensor],
                                training: bool = False) -> Dict[AgentID, torch.Tensor]:
        """
        Creates a dictionary of incoming messages for each agent given a
        dictionary of outgoing messages from each agent.

        Args:
            msgs_out: A dictionary where each key is a sending agent's id, and each
                value is a tensor of `batch_size x n_agents x message_dim`, with
                each row at index i of the tensor being a message from the key agent
                for the agent at index i .
            training: A boolean representing whether or not inference is running
                in training mode. Passed to the channel function.
    
        Returns:
            A dictionary where each key is a receiving agent's id, and each row
            is a is a tensor of `batch_size x n_agents x message_dim`, with
            each row at index i of the tensor being a message for the key agent
            from the agent at index i.
        """
        agent_ids = list(msgs_out.keys())

        # mask outgoing messages that are not permitted by the comm spec
        for sender_agent in agent_ids:
            batch_size, *_ = msgs_out[sender_agent].shape
            mask = self.get_comm_mask(sender_agent, batch_size)
            mask = mask.to(msgs_out[sender_agent].device)
            msgs_out[sender_agent] = mask * msgs_out[sender_agent]

        # create tensors of incoming messages for each receiver
        msgs_in = {}
        for receiver_agent in agent_ids:
            # msgs at index i of msgs_in[receiver_agent] are the messages
            # received by receiver_agent from agent i
            i = self.comms_spec.get_agent_idx(receiver_agent)
            # collect messages from other agents
            msgs = torch.cat([
                msgs_out[sender_agent][:, i:i + 1]
                for sender_agent in agent_ids
            ], dim=-2)
            msgs_in[receiver_agent] = self.comm_channel_fn(msgs, training=training)

        return msgs_in

    def _actors_forward(self,
                        batch: NestedDict[Any],
                        training: bool = False):
        msgs_out = {}
        encodings = {}
        for agent_id in self.get_agent_ids():
            obs_enc, msgs = self.encode_obs_and_produce_msgs(agent_id, batch[agent_id])
            encodings[agent_id] = obs_enc
            msgs_out[agent_id] = msgs

        msgs_in = self._handle_message_passing(msgs_out, training=training)

        outputs = {}
        for agent_id, obs_enc in encodings.items():
            action_head = self.agent_models[agent_id][ACTION_HEAD]
            outputs[agent_id] = {
                SampleBatch.ACTION_DIST_INPUTS: action_head(obs_enc, msgs_in[agent_id])
            }

        self.last_msgs_sent = msgs_in 
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
        return self._common_forward(batch, mode=INFERENCE_FORWARD)

    @override(RLModule)
    def _forward_exploration(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch, mode=EXPLORATION_FORWARD)

    @override(RLModule)
    def _forward_train(self, batch: NestedDict) -> Mapping[str, Any]:
        return self._common_forward(batch, mode=TRAIN_FORWARD)

    def _common_forward(self, batch: NestedDict[Any], mode: str) -> Mapping[str, Any]:
        """
        Common forward pass for all modes.
        """
        not_inference_mode = mode != INFERENCE_FORWARD
        self.last_inputs = batch

        outputs = {}
        actor_inputs = self.nest_by_agent(batch)
        actor_outputs = self._actors_forward(actor_inputs,
                                             training=not_inference_mode)

        outputs[SampleBatch.ACTION_DIST_INPUTS] = torch.cat([
            outs[SampleBatch.ACTION_DIST_INPUTS]
            for outs in actor_outputs.values()
        ], axis=-1)

        self.last_actor_outputs = actor_outputs

        if not_inference_mode:
            outputs[SampleBatch.VF_PREDS] = self._critics_forward(batch)

        return outputs

    def named_parameters(self):
        params = list(super().named_parameters())
        if self.share_actor_params:
            models, *_ = self.agent_models.values()
            for model in models.values():
                params += list(model.named_parameters())
        else:
            for agent_id, models in self.agent_models.items():
                for model_id, model in models.items():
                    params += [
                        (f'{agent_id}/{model_id}/{name}', param)
                        for name, param in model.named_parameters()
                    ]
        return params

    def parameters(self):
        return [param for _, param in self.named_parameters()]

    def update_hyperparameters(self, iteration: int) -> dict:
        new_hyperparams = {}
        new_hyperparams.update(self.comm_channel_fn.update_hyperparams(iteration))
        return new_hyperparams

    def get_last_msgs(self,
                      batch_idx: Optional[int] = None,
                      return_np: bool = True) -> Dict[str, Dict[str, Union[np.ndarray, torch.Tensor]]]:
        """
        Constructs a dictionary of messages sent between agents
        in the last forward pass.

        The dictionary is of the form:
        {
            receiver_id: {
                sender_id: msg
            }
        }

        Args:
            batch_index: The index of the batch to get the messages from.

        Returns:
            The messages network dictionary.
        """
        def get_msgs(sent_msgs_batch, sender_id) -> Union[np.ndarray, torch.Tensor]:
            sender_idx = self.comms_spec.get_agent_idx(sender_id)
            if batch_idx is None:
                msgs = sent_msgs_batch[:, sender_idx]
            else:
                msgs = sent_msgs_batch[batch_idx, sender_idx]

            if return_np:
                return msgs.detach().cpu().numpy()

            return msgs

        if self.last_msgs_sent is not None:
            return {
                receiver_id: {
                    sender_id: get_msgs(msgs_batch, sender_id)
                    for sender_id in self.comms_spec.comm_channels
                    if self.comms_spec.can_send(receiver_id, sender_id)
                }
                for receiver_id, msgs_batch in self.last_msgs_sent.items()
            }
        else:
            return None
