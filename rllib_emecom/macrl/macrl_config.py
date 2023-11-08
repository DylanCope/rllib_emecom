from .comms.comms_spec import CommunicationSpec, CommNetwork

from typing import Any, List, Optional

from ray.rllib.algorithms.algorithm_config import NotProvided


def get_fully_connected_comm_channels(agent_ids: List[Any]) -> CommNetwork:
    """
    Creates a fully connected communication network, so each agent can
    send and receive messages from every other agent (excluding itself).
    """
    return {
        agent_id: [
            other_id for other_id in agent_ids
            if other_id != agent_id
        ]
        for agent_id in agent_ids
    }


class MACRLConfig:

    def __init__(self):
        self.comm_spec: CommunicationSpec = None
        self.message_dim = 16
        self.comm_channels = None
        self.channel_fn = 'straight_through'
        self.channel_fn_config = {}

    def communications(self,
                       comm_spec: Optional[CommunicationSpec] = NotProvided,
                       message_dim: Optional[int] = NotProvided,
                       agent_ids: Optional[List[Any]] = NotProvided,
                       comm_channels: Optional[CommNetwork] = NotProvided,
                       channel_fn: Optional[str] = NotProvided,
                       channel_fn_config: Optional[dict] = NotProvided,) -> 'MACRLConfig':
        if comm_spec is not NotProvided:
            self.message_dim = comm_spec.message_dim
            self.comm_channels = comm_spec.comm_channels
            self.channel_fn = comm_spec.channel_fn
            self.channel_fn_config = comm_spec.channel_fn_config

        else:
            if comm_channels is NotProvided or comm_channels is None:
                assert agent_ids is not NotProvided, \
                    'Agent ids must be provided if comm_spec and '\
                    'comm_channels are not provided'
                self.comm_channels = get_fully_connected_comm_channels(agent_ids)
            else:
                self.comm_channels = comm_channels

            if message_dim is not NotProvided and message_dim is not None:
                self.message_dim = message_dim

            if channel_fn is not NotProvided and channel_fn is not None:
                self.channel_fn = channel_fn

            if channel_fn_config is not NotProvided and channel_fn_config is not None:
                self.channel_fn_config = channel_fn_config

        return self

    def create_comm_spec(self) -> CommunicationSpec:
        return CommunicationSpec(
            message_dim=self.message_dim,
            comm_channels=self.comm_channels,
            channel_fn=self.channel_fn,
            channel_fn_config=self.channel_fn_config
        )

    def build(self):
        self.comm_spec = self.create_comm_spec()

    def validate(self):
        assert self.message_dim > 0, \
            'Message dimension must be positive'
        assert self.comm_channels is not None, \
            'Communication channels must be provided'

    def to_dict(self):
        return {
            'message_dim': self.message_dim,
            'comm_channels': self.comm_channels,
            'channel_fn': self.channel_fn,
            'channel_fn_config': self.channel_fn_config
        }

    def update_from_dict(self, config_dict: dict) -> 'MACRLConfig':
        if 'message_dim' in config_dict:
            self.message_dim = config_dict['message_dim']
        if 'comm_channels' in config_dict:
            self.comm_channels = config_dict['comm_channels']
        if 'channel_fn' in config_dict:
            self.channel_fn = config_dict['channel_fn']
        if 'channel_fn_config' in config_dict:
            self.channel_fn_config = config_dict['channel_fn_config']
        return self
