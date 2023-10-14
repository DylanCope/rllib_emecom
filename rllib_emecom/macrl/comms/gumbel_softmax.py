from .comm_channel_fn import CommunicationChannelFunction

from typing import Optional
import numpy as np

import torch
from torch.distributions.relaxed_categorical import RelaxedOneHotCategorical
from torch.distributions.one_hot_categorical import OneHotCategorical


class GumbelSoftmaxCommunicationChannel(CommunicationChannelFunction):

    def __init__(self,
                 temperature: float = 1.0,
                 temperature_annealing: bool = False,
                 annealing_start_iter: int = 0,
                 n_anneal_iterations: Optional[int] = None,
                 final_temperature: Optional[float] = None,
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = temperature

        self.temperature_annealing = temperature_annealing
        self.n_anneal_iterations = n_anneal_iterations
        self.final_temperature = final_temperature
        self.annealing_start_iter = annealing_start_iter
        if self.temperature_annealing:
            assert self.n_anneal_iterations is not None and \
                self.final_temperature is not None, \
                "If annealing the temperature, the number of annealing iterations " \
                "and the final temperature must be specified."

            self.start_temperature = temperature
            self.annealing_k = -np.log(final_temperature / temperature) / n_anneal_iterations

    def call(self, message: torch.Tensor,
             training: bool = False) -> torch.Tensor:
        if training:
            return RelaxedOneHotCategorical(self.temperature,
                                            logits=message).rsample()
        else:
            return OneHotCategorical(logits=message).sample()

    def anneal_temperature(self, iteration: int):
        if self.annealing_start_iter < iteration < self.n_anneal_iterations:
            x = max(iteration - self.annealing_start_iter, 0)
            anneal_factor = np.exp(-self.annealing_k * x)
            self.temperature = float(self.start_temperature * anneal_factor)

        elif iteration >= self.n_anneal_iterations:
            self.temperature = self.final_temperature

    def update_hyperparams(self, iteration: int) -> dict:
        if self.temperature_annealing:
            self.anneal_temperature(iteration)
        return {
            'comm_channel_temperature': self.temperature
        }
