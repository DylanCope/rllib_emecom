from rllib_emecom.tests.test_macrl_module import create_test_macrl_module

import argparse
from pathlib import Path

import torch
from torchview import draw_graph
from ray.rllib.core.models.base import SampleBatch


def draw_macrl_graph(output_name: str, graph_depth: int = 3, **kwargs):
    macrl_module, _ = create_test_macrl_module(**kwargs)
    obs_space = macrl_module.config.observation_space
    batch_size = 4
    batch = {
        SampleBatch.OBS: torch.vstack([
            torch.tensor(obs_space.sample())
            for _ in range(batch_size)
        ]),
    }

    graph = draw_graph(macrl_module, {'batch': batch},
                       depth=graph_depth, device='cpu', expand_nested=True)
    graph.visual_graph.render(output_name, format='pdf', cleanup=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--graph_depth', type=int, default=3)
    cmd_args = parser.parse_args()

    outputs_dir = cmd_args.output_dir or 'test/macrl_graphs'
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
                             graph_depth=cmd_args.graph_depth,
                             n_agents=2, channel_fn=channel_fn, **channel_args)
