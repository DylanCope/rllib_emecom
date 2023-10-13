from abc import ABC

import torch


class CommunicationChannelFunction(ABC):

    def __init__(self, **unused_channel_config) -> None:
        self.unused_channel_config = unused_channel_config
        self.force_eval_mode = False

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def set_force_eval(self, force_eval: bool) -> None:
        self.force_eval_mode = force_eval

    def __call__(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        training = training and not self.force_eval_mode
        return self.call(message, training)

    def update_hyperparams(self, iteration: int) -> dict:
        return {}
