from .comm_channel_fn import CommunicationChannelFunction

import torch


class StraightThroughCommunicationChannel(CommunicationChannelFunction):

    def call(self, message: torch.Tensor,
             training: bool = False) -> torch.Tensor:
        return message
