from abc import ABC

import torch


class CommunicationChannelFunction(ABC):

    def __init__(self, **unused_channel_config) -> None:
        self.unused_channel_config = unused_channel_config

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def __call__(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        return self.call(message, training)

    def update_hyperparams(self, iteration: int) -> dict:
        return {}
