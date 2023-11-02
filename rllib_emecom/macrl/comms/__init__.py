from .comm_channel_fn import CommunicationChannelFunction
from .straight_through import StraightThroughCommunicationChannel
from .gumbel_softmax import GumbelSoftmaxCommunicationChannel
from .dru import DiscretiseRegulariseCommunicationChannel

from difflib import get_close_matches


_CHANNEL_FNS_REGISTRY = {
    'straight_through': StraightThroughCommunicationChannel,
    'gumbel_softmax': GumbelSoftmaxCommunicationChannel,
    'dru': DiscretiseRegulariseCommunicationChannel,
    'discrete_regularise': DiscretiseRegulariseCommunicationChannel,
}


def get_channel_fn_cls(channel_fn: str) -> CommunicationChannelFunction:
    if channel_fn.lower() not in _CHANNEL_FNS_REGISTRY:
        close_matches = ', '.join(get_close_matches(channel_fn.lower(),
                                                    _CHANNEL_FNS_REGISTRY.keys()))
        raise NotImplementedError(
            f"Channel function {channel_fn} not found. "
            f"Did you mean one of the following: {close_matches}"
        )

    return _CHANNEL_FNS_REGISTRY[channel_fn.lower()]


def register_channel_fn(channel_fn: str,
                        channel_fn_cls: CommunicationChannelFunction) -> None:
    _CHANNEL_FNS_REGISTRY[channel_fn.lower()] = channel_fn_cls
