# from typing import Any, Mapping

from abc import ABC
from functools import cached_property
from typing import Any, List, Dict, Optional

from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CommunicationChannelFunction(ABC):

    def __init__(self, **unused_channel_config) -> None:
        self.unused_channel_config = unused_channel_config

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.call(message, training)


class StraightThroughCommunicationChannel(CommunicationChannelFunction):

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        return message


class GumbelSoftmaxCommunicationChannel(CommunicationChannelFunction):

    def __init__(self, temperature: float = 1.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        # return torch.nn.functional.gumbel_softmax(message, tau=self.temperature, hard=True)
        if training:
            return RelaxedOneHotCategorical(self.temperature, logits=message).rsample()
        else:
            return OneHotCategorical(logits=message).sample()


class DiscreteRegulariseCommunicationChannel(CommunicationChannelFunction):

    def __init__(self, channel_noise: float = 0.5, channel_activation='softmax', **kwargs) -> None:
        super().__init__(**kwargs)
        self.channel_noise = channel_noise
        self.channel_activation = channel_activation

    def softmax_forward(self, z: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            z = torch.nn.functional.softmax(z, dim=-1)
            return z
        msgs_symbs = torch.argmax(z, dim=-1)
        n_msgs = z.shape[-1]
        return torch.nn.functional.one_hot(msgs_symbs, n_msgs)

    def sigmoid_forward(self, z: torch.Tensor, training: bool) -> torch.Tensor:
        z = torch.nn.functional.sigmoid(z)
        if training:
            return z
        return torch.round(z)

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        if training:
            z = message + torch.randn_like(message) * self.channel_noise
        else:
            z = message

        if self.channel_activation == 'softmax':
            return self.softmax_forward(z, training)

        elif self.channel_activation == 'sigmoid':
            return self.sigmoid_forward(z, training)

        raise NotImplementedError("Unknown channel activation function: " + self.channel_activation)


class CommunicationNetwork(ABC):

    def __init__(self,
                 message_dim: int,
                 channel_fn: Optional[CommunicationChannelFunction] = None) -> None:
        self.message_dim = message_dim
        self.messages = {}
        self.to_send = {}
        self.channel_fn = channel_fn or StraightThroughCommunicationChannel()

    def send_message(self,
                     message: torch.Tensor,
                     agent_to: str,
                     agent_from: str,
                     training: bool = False) -> None:
        if agent_to not in self.to_send:
            self.to_send[agent_to] = {}
        self.to_send[agent_to][agent_from] = self.channel_fn(message, training)

    def tick(self) -> None:
        self.messages = self.to_send
        self.to_send = {}

    def set_state(self, messages: dict):
        self.messages = messages
        self.to_send = {}

    def has_incoming_messages(self, agent_id: str) -> bool:
        return agent_id in self.messages

    def get_incoming_messages(self, agent_id: str) -> torch.Tensor:
        return torch.stack(list(self.messages[agent_id].values()))

    def can_send(self, agent_to: str, agent_from: str) -> bool:
        raise NotImplementedError

    def agents_can_send_to(self, agent_from: str) -> List[str]:
        raise NotImplementedError

    def agents_can_receive_from(self, agent_to: str) -> List[str]:
        raise NotImplementedError


class StaticCommunicationNetwork(CommunicationNetwork):

    def __init__(self,
                 message_dim: int,
                 comm_channels: Dict[str, List[str]],
                 channel_fn: Optional[CommunicationChannelFunction] = None) -> None:
        """
        A communication network that is static, i.e. the communication channels are fixed.

        Args:
            message_dim: The dimension of the messages.
            comm_channels: A dictionary mapping agent ids to a list of agent ids
                that they can send messages to.
            channel_fn: The function that is applied to the messages before they are sent.
        """
        super().__init__(message_dim, channel_fn)
        self.comm_channels = comm_channels

    def can_send(self, agent_to: str, agent_from: str) -> bool:
        return agent_to in self.comm_channels[agent_from]

    def agents_can_send_to(self, agent_from: str) -> List[str]:
        return self.comm_channels.get(agent_from, [])

    def agents_can_receive_from(self, agent_to: str) -> List[str]:
        return [agent_from for agent_from, agent_tos in self.comm_channels.items()
                if agent_to in agent_tos]


class CommunicationSpec(dict):
    """
    A specification for construting a Communication Network.

    Args:
        message_dim: The dimension of the messages.
        n_agents: The number of agents in the environment.
        comm_channels: A dictionary mapping agent ids to a list of agent ids.
            The keys are the sender agents and the values are the receiver agents.
        static: Whether the communication network is static or not.
            Only static networks are supported currently.
        channel_fn: The function that is applied to the messages before they are sent.
            Only "straight_through" and "gumbel_softmax" supported currently.
    """

    def __init__(self,
                 message_dim: int,
                 comm_channels: Dict[Any, List[Any]],
                 static: bool = True,
                 channel_fn: str = "straight_through",
                 channel_fn_config: dict = None):
        self.message_dim = message_dim
        self.comm_channels = comm_channels
        self.static = static
        self.channel_fn = channel_fn
        self.channel_fn_config = channel_fn_config or {}

        self.validate()

        dict.__init__(self, **self.to_dict())

    def can_send(self, agent_to: Any, agent_from: Any) -> bool:
        return agent_to in self.comm_channels[agent_from]

    @cached_property
    def agents(self) -> List[Any]:
        return sorted(list(set(
            list(self.comm_channels.keys())
            + sum(self.comm_channels.values(), [])
        )))

    @cached_property
    def n_agents(self) -> int:
        return len(self.agents)

    @cached_property
    def agent_to_index(self) -> Dict[Any, int]:
        return {
            agent: i for i, agent in enumerate(self.agents)
        }

    def index_to_agent(self, idx: int) -> Any:
        return self.agents[idx]

    def get_agent_idx(self, agent: Any) -> int:
        return self.agent_to_index[agent]

    def validate(self) -> None:
        if self.n_agents < 1:
            raise ValueError("The number of agents must be at least 1.")

    def get_channel_fn_cls(self):
        if self.channel_fn.lower() == "straight_through":
            return StraightThroughCommunicationChannel
        elif self.channel_fn.lower() == "gumbel_softmax":
            return GumbelSoftmaxCommunicationChannel
        elif self.channel_fn.lower() in ["discrete_regularise", "dru"]:
            return DiscreteRegulariseCommunicationChannel
        else:
            raise NotImplementedError(
                f"Channel function {self.channel_fn} not implemented. "
                "Only 'straight_through', 'dru' and 'gumbel_softmax' are currently supported."
            )

    def get_channel_fn(self) -> CommunicationChannelFunction:
        return self.get_channel_fn_cls()(**self.channel_fn_config)

    def build(self) -> CommunicationNetwork:
        if self.static:
            return StaticCommunicationNetwork(
                message_dim=self.message_dim,
                comm_channels=self.comm_channels,
                channel_fn=self.get_channel_fn()
            )
        else:
            raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_dim": self.message_dim,
            "comm_channels": self.comm_channels,
            "static": self.static,
            "channel_fn": self.channel_fn,
            "channel_fn_config": self.channel_fn_config,
            "n_agents": self.n_agents
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunicationSpec":
        return CommunicationSpec(
            message_dim=d["message_dim"],
            comm_channels=d["comm_channels"],
            static=d["static"],
            channel_fn=d["channel_fn"],
            channel_fn_config=d["channel_fn_config"],
            n_agents=d["n_agents"]
        )
