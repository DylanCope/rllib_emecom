from .test_macrl_module import create_mock_macrl_module

from pathlib import Path

import torch
from torchview import draw_graph
from ray.rllib.core.models.base import SampleBatch


def draw_macrl_graph(output_name: str, **kwargs):
    macrl_module, _ = create_mock_macrl_module(**kwargs)
    obs_space = macrl_module.config.observation_space
    batch_size = 4
    batch = {
        SampleBatch.OBS: torch.vstack([
            torch.tensor(obs_space.sample())
            for _ in range(batch_size)
        ]),
    }

    graph = draw_graph(macrl_module, {'batch': batch}, device='cpu', expand_nested=True)
    graph.visual_graph.render(output_name, format='pdf', cleanup=True)


if __name__ == '__main__':
    outputs_dir = 'test/macrl_graphs'
    Path(outputs_dir).mkdir(exist_ok=True)

    args = {
        'straight_through': [
            {}
        ],
        'dru': [
            {'comm_channel_noise': 0.0},
            {'comm_channel_noise': 0.5},
        ],
        'gumbel_softmax': [
            {},
        ],
    }

    for channel_fn in ['straight_through', 'dru', 'gumbel_softmax']:
        for i, channel_args in enumerate(args[channel_fn]):
            draw_macrl_graph(f'{outputs_dir}/macrl_module_{channel_fn}_{i}',
                             n_agents=2, channel_fn=channel_fn, **channel_args)
