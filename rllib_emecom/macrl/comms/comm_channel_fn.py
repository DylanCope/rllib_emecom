from abc import ABC

import torch


class CommunicationChannelFunction(ABC):

    def __init__(self, **unused_channel_config) -> None:
        self.unused_channel_config = unused_channel_config
        self.force_eval_mode = False
        self.comms_disabled = False

    def call(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        raise NotImplementedError

    def set_force_eval(self, force_eval: bool) -> None:
        self.force_eval_mode = force_eval

    def disable(self) -> None:
        self.comms_disabled = True

    def enable(self) -> None:
        self.comms_disabled = False

    def __call__(self, message: torch.Tensor, training: bool = False) -> torch.Tensor:
        if self.comms_disabled:
            return torch.zeros_like(message).to(message.device)
        training = training and not self.force_eval_mode
        return self.call(message, training)

    def update_hyperparams(self, iteration: int) -> dict:
        return {}
