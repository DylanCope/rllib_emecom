from argparse import Namespace
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.utils.nested_dict import NestedDict
from ray.rllib.utils.metrics import ALL_MODULES
from ray.rllib.core.learner.torch.torch_learner import TorchLearner
from ray.rllib.execution.rollout_ops import (
    standardize_fields,
    synchronous_parallel_sample,
)
from ray.rllib.utils.minibatch_utils import MiniBatchCyclicIterator

import torch


def gradients_summary_barplot(gradients_df: pd.DataFrame, label='norm', ax=None):
    if ax is None:
        _, ax = plt.subplots()

    # sns.barplot(x=list(gradients_dict.keys()), y=gradient_norms, ax=ax)
    sns.barplot(data=gradients_df[gradients_df.stat == label], x='name', y='grad', ax=ax)
    ax.set_xlabel('Parameter')
    ax.set_ylabel(f'Gradient {label}')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45,
                       horizontalalignment='right')
    return ax


def get_learner(algo: Algorithm) -> TorchLearner:
    assert algo.learner_group._is_local
    learner = algo.learner_group._learner
    assert isinstance(learner, TorchLearner)
    return learner


def get_gradients(algo: Algorithm,
                  batch: NestedDict,
                  postprocess: bool = True) -> Dict[str, torch.Tensor]:
    learner = get_learner(algo)
    fwd_out = learner.module.forward_train(batch)
    loss_per_module = learner.compute_loss(fwd_out=fwd_out, batch=batch)

    # same logic as learner.compute_gradients() but gradients are given names

    for optim in learner._optimizer_parameters:
        # set_to_none is a faster way to zero out the gradients
        optim.zero_grad(set_to_none=True)

    loss_per_module[ALL_MODULES].backward()

    def get_param_name(module_name, name):
        if len(learner.module._rl_modules) == 1:
            return name
        return f'{module_name}/{name}'

    gradients = {
        get_param_name(module_name, name): param.grad
        for module_name, module in learner.module._rl_modules.items()
        for name, param in module.named_parameters()
    }

    if postprocess:
        for param_refs in learner._optimizer_parameters.values():
            param_refs.extend(gradients.keys())
        gradients = learner.postprocess_gradients(gradients)

    gradients = {
        name: grad.detach().cpu().numpy()
        for name, grad in gradients.items()
    }

    return gradients


def sample_batch(algo: Algorithm, batch_size: int) -> NestedDict:
    minibatch_size = batch_size
    num_iters = 1
    batch = synchronous_parallel_sample(
        worker_set=algo.workers, max_env_steps=batch_size
    )
    batch = batch.as_multi_agent()
    batch = standardize_fields(batch, ["advantages"])
    learner = get_learner(algo)
    batch = learner._convert_batch_type(batch)
    batch = learner._set_slicing_by_batch_id(batch, value=True)
    tensor_minibatch = next(iter(MiniBatchCyclicIterator(batch, minibatch_size, num_iters)))
    return NestedDict(tensor_minibatch.policy_batches)


def get_grads_summaries_dict(gradients_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    summary_fns = [
        ('Norm', np.linalg.norm),
        ('Max', np.max),
        ('Min', np.min),
    ]
    return pd.DataFrame([
        {'name': name, 'grad': stat_fn(g), 'stat': stat}
        for stat, stat_fn in summary_fns
        for name, g in gradients_dict.items()
    ])


def create_and_save_gradients_plot(args: Namespace, output: str):
    config = get_goal_comms_config(args)
    algo = config.build()
    batch = sample_batch(algo, 512)
    gradients = get_gradients(algo, batch)

    df = get_grads_summaries_dict(gradients)
    stat_types = df.stat.unique()
    n_stats = len(stat_types)

    sns.set()
    _, axs = plt.subplots(1, n_stats, figsize=(12 * n_stats, 8))
    for stat, ax in zip(stat_types, axs):
        gradients_summary_barplot(df, stat, ax=ax)
    plt.savefig(output, bbox_inches='tight')
    print(f'Saved figure to {output}')
    plt.close()


def create_straight_through_plots(args: Namespace, output_dir: str):
    output_fig = f'{output_dir}/gradients.pdf'
    create_and_save_gradients_plot(args, output_fig)


def create_dru_plots(args: Namespace, output_dir: str):
    for noise in [0.0, 0.1, 0.2, 0.5, 1.0]:
        args.comm_channel_noise = noise
        activation = args.comm_channel_activation
        output_fig = f'{output_dir}/gradients-{activation=}-{noise=}.pdf'
        create_and_save_gradients_plot(args, output_fig)


def create_gumbel_softmax_plots(args: Namespace, output_dir: str):
    for temp in [0.1, 0.2, 0.5, 1.0]:
        args.comm_channel_temp = temp
        output_fig = f'{output_dir}/gradients-{temp=}.pdf'
        create_and_save_gradients_plot(args, output_fig)


if __name__ == '__main__':
    from rllib_emecom.train.goal_comms import get_goal_comms_config, parse_args
    from rllib_emecom.utils.experiment_utils import initialise_ray
    import ray
    from pathlib import Path

    initialise_ray()

    try:
        args = parse_args()
        output_dir = f'analysis/comm_channel_gradients/{args.comm_channel_fn}'
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if args.comm_channel_fn == 'straight_through':
            create_straight_through_plots(args, output_dir)
        elif args.comm_channel_fn == 'dru':
            create_dru_plots(args, output_dir)
        elif args.comm_channel_fn == 'gumbel_softmax':
            create_gumbel_softmax_plots(args, output_dir)
        else:
            raise NotImplementedError(
                f'Unknown comm channel fn: {args.comm_channel_fn}')

    finally:
        ray.shutdown()
