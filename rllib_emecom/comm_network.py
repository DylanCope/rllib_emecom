# from typing import Any, Mapping

from abc import ABC
from functools import cached_property
from typing import Any, List, Dict, Optional

from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CommunicationMode:
    TRAINING = 'training'
    EXPLORATION = 'exploration'
    INFERENCE = 'inference'

    @staticmethod
    def from_forward_fn_name(forward_fn: str):
        if forward_fn == 'train_forward':
            return CommunicationMode.TRAINING
        if forward_fn == 'exploration_forward':
            return CommunicationMode.EXPLORATION
        if forward_fn == 'inference_forward':
            return CommunicationMode.INFERENCE


class CommunicationChannelFunction(ABC):

    def call(self, message: torch.Tensor, mode: str = CommunicationMode.TRAINING) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, message: torch.Tensor, mode: str = CommunicationMode.TRAINING) -> torch.Tensor:
        return self.call(message, mode)


class StraightThroughCommunicationChannel(CommunicationChannelFunction):

    def call(self, message: torch.Tensor, mode: str = CommunicationMode.TRAINING) -> torch.Tensor:
        return message


class CommunicationNetwork(nn.Module, ABC):

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
                     mode: str = CommunicationMode.TRAINING) -> None:
        if agent_to not in self.to_send:
            self.to_send[agent_to] = {}
        self.to_send[agent_to][agent_from] = self.channel_fn(message, mode)

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
            Only straight-through is supported currently.
    """

    def __init__(self,
                 message_dim: int,
                 comm_channels: Dict[Any, List[Any]],
                 static: bool = True,
                 channel_fn: str = "straight_through"):
        self.message_dim = message_dim
        self.comm_channels = comm_channels
        self.static = static
        self.channel_fn = channel_fn

        self.validate()

        dict.__init__(self, **self.to_dict())

    @cached_property
    def agents(self) -> List[Any]:
        return sorted(list(set(
            list(self.comm_channels.keys()) + sum(self.comm_channels.values(), [])
        )))

    @cached_property
    def n_agents(self) -> int:
        return len(self.agents)

    @cached_property
    def agent_to_index(self) -> Dict[Any, int]:
        return {agent: i for i, agent in enumerate(self.agents)}

    def index_to_agent(self, idx: int) -> Any:
        return self.agents[idx]

    def get_agent_idx(self, agent: Any) -> int:
        return self.agent_to_index[agent]

    def validate(self) -> None:
        if self.n_agents < 1:
            raise ValueError("The number of agents must be at least 1.")

    def build(self) -> CommunicationNetwork:
        if self.static:

            if self.channel_fn is None or self.channel_fn == "straight_through":
                channel_fn = StraightThroughCommunicationChannel()
            else:
                raise NotImplementedError

            return StaticCommunicationNetwork(
                message_dim=self.message_dim,
                comm_channels=self.comm_channels,
                channel_fn=channel_fn
            )
        else:
            raise NotImplementedError

    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_dim": self.message_dim,
            "comm_channels": self.comm_channels,
            "static": self.static,
            "channel_fn": self.channel_fn,
            "n_agents": self.n_agents
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommunicationSpec":
        return CommunicationSpec(
            message_dim=d["message_dim"],
            comm_channels=d["comm_channels"],
            static=d["static"],
            channel_fn=d["channel_fn"],
            n_agents=d["n_agents"]
        )
