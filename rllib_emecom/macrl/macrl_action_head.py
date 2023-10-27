from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec

from gymnasium.spaces import Discrete
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog, _check_if_diag_gaussian
from ray.rllib.core.models.configs import MLPHeadConfig, FreeLogStdMLPHeadConfig
from ray.rllib.core.models.catalog import MODEL_DEFAULTS

torch, nn = try_import_torch()


DEFAULT_AGGREGATOR_CONFIG = {
    **MODEL_DEFAULTS,
    **{
        'fcnet_hiddens': [256,],
        'fcnet_activation': 'relu',
    }
}


class MACRLActionHead(nn.Module):
    """
    MACRL action head takes an encoding of the inputs for an agent,
    and the messages for that agent from the other agents, and returns
    inputs to the agent's action distribution.
    """
    framework: str = "torch"

    def __init__(self,
                 action_space: Discrete,
                 actor_encoding_dim: int,
                 catalog: PPOCatalog,
                 comms_spec: CommunicationSpec):
        super().__init__()
        self.action_space = action_space
        self.catalog = catalog
        self.comms_spec = comms_spec
        self.n_agents = comms_spec.n_agents
        self.message_dim = comms_spec.message_dim
        self.aggregate_msgs_and_obs = \
            self.build_msg_and_obs_aggregator(actor_encoding_dim)
        self.pi_head = self.build_pi_head()

    def build_pi_head(self) -> nn.Module:
        # Get action_distribution_cls to find out about the
        # output dimension for pi_head
        action_distribution_cls = \
            self.catalog.get_action_dist_cls(framework=self.framework)

        if self.catalog._model_config_dict["free_log_std"]:
            _check_if_diag_gaussian(
                action_distribution_cls=action_distribution_cls,
                framework=self.framework
            )

        assert isinstance(self.action_space, Discrete), \
            "Expected action space to be a Discrete."

        required_output_dim = int(self.action_space.n)
        # Now that we have the action dist class and number of
        # outputs, we can define our pi-config and build the pi head.
        pi_head_config_class = (
            FreeLogStdMLPHeadConfig
            if self.model_config["free_log_std"]
            else MLPHeadConfig
        )

        pi_and_vf_head_hiddens = \
            self.model_config["post_fcnet_hiddens"]
        pi_and_vf_head_activation = \
            self.model_config["post_fcnet_activation"]

        pi_head_config = pi_head_config_class(
            input_dims=self.catalog.latent_dims,
            hidden_layer_dims=pi_and_vf_head_hiddens,
            hidden_layer_activation=pi_and_vf_head_activation,
            output_layer_dim=required_output_dim,
            output_layer_activation="linear",
        )

        return pi_head_config.build(framework=self.framework)

    @property
    def model_config(self):
        return self.catalog._model_config_dict

    def build_msg_and_obs_aggregator(self,
                                     actor_encoding_dim: int) -> nn.Module:
        aggregator_config = {
            **DEFAULT_AGGREGATOR_CONFIG,
            **self.model_config.get('msg_obs_aggregator', dict()),
        }

        layers = []
        dim_in = self.n_agents * self.message_dim + actor_encoding_dim
        for dim_out in aggregator_config['fcnet_hiddens']:
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            dim_in = dim_out

        return nn.Sequential(*layers)

    def forward(self,
                obs_encoding: torch.Tensor,
                msgs_in: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_encoding: agent's observation encoding
            msgs_in: a tensor with dims `batch_size x n_agents x message_dim`
                that contains messages from each of the agents in the env

        Returns:
            A tensor with dims `batch_size x n_actions` of action logits
        """
        device = obs_encoding.device
        self.aggregate_msgs_and_obs.to(device)
        self.pi_head.to(device)

        agent_msgs_in = msgs_in.view(-1, self.n_agents * self.message_dim)
        # aggregator_inp = {
        #     SampleBatch.OBS: torch.cat([agent_msgs_in, obs_encoding], dim=-1)
        # }
        # final_encoding = self.aggregate_msgs_and_obs(aggregator_inp)[ENCODER_OUT]
        aggregator_inp = torch.cat([agent_msgs_in, obs_encoding], dim=-1)
        final_encoding = self.aggregate_msgs_and_obs(aggregator_inp)
        return self.pi_head(final_encoding)
