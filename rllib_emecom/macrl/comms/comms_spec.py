from .comm_channel_fn import CommunicationChannelFunction
from .straight_through import StraightThroughCommunicationChannel
from .gumbel_softmax import GumbelSoftmaxCommunicationChannel
from .dru import DiscretiseRegulariseCommunicationChannel

from functools import cached_property
from typing import Any, List, Dict
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()


class CommunicationSpec(dict):
    """
    A specification for construting a Communication Network.

    Args:
        message_dim: The dimension of the messages.
        n_agents: The number of agents in the environment.
        comm_channels: A dictionary mapping agent ids to a list of agent ids.
            The keys are the sender agents and the values are the receiver agents.
        channel_fn: The function that is applied to the messages before they are sent.
            Only "straight_through" and "gumbel_softmax" supported currently.
    """

    def __init__(self,
                 message_dim: int,
                 comm_channels: Dict[Any, List[Any]],
                 channel_fn: str = "straight_through",
                 channel_fn_config: dict = None):
        self.message_dim = message_dim
        self.comm_channels = comm_channels
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
            return DiscretiseRegulariseCommunicationChannel
        else:
            raise NotImplementedError(
                f"Channel function {self.channel_fn} not implemented. "
                "Only 'straight_through', 'dru' and 'gumbel_softmax' are currently supported."
            )

    def get_channel_fn(self) -> CommunicationChannelFunction:
        return self.get_channel_fn_cls()(**self.channel_fn_config)

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
