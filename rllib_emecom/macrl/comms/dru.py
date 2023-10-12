from .comm_channel_fn import CommunicationChannelFunction

import torch


class DiscretiseRegulariseCommunicationChannel(CommunicationChannelFunction):

    def __init__(self,
                 channel_noise: float = 0.5,
                 channel_activation='softmax',
                 **kwargs) -> None:
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

    def call(self, message: torch.Tensor,
             training: bool = False) -> torch.Tensor:
        if training:
            z = message + torch.randn_like(message) * self.channel_noise
        else:
            z = message

        if self.channel_activation == 'softmax':
            return self.softmax_forward(z, training)

        elif self.channel_activation == 'sigmoid':
            return self.sigmoid_forward(z, training)

        raise NotImplementedError("Unknown channel activation function: "
                                  + self.channel_activation)
