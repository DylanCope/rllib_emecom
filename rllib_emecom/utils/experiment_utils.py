from rllib_emecom.utils.register_envs import register_envs

import re
from pathlib import Path
from typing import List, Optional
from argparse import Namespace
import pandas as pd

import ray
from ray import tune
from ray.train import Checkpoint
from ray.air import CheckpointConfig
from ray.tune.analysis.experiment_analysis import NewExperimentAnalysis
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig

WANDB_PROJECT = 'rllib_emecom'


def initialise_ray(dashboard_host='0.0.0.0',
                   dashboard_port=8265,
                   ignore_reinit_error=True,
                   **init_kwargs):
    ray.init(dashboard_host=dashboard_host,
             dashboard_port=dashboard_port,
             ignore_reinit_error=ignore_reinit_error,
             **init_kwargs)
    register_envs()


def get_wandb_callback(project: Optional[str] = None) -> WandbLoggerCallback:
    project = project or WANDB_PROJECT
    if Path('.wandb_api_key').exists():
        return WandbLoggerCallback(project=project, api_key_file='.wandb_api_key')
    return WandbLoggerCallback(project=project)


def run_training(config: AlgorithmConfig, args: Namespace):
    try:
        initialise_ray()

        callbacks = []
        if not args.no_wandb:
            callbacks.append(get_wandb_callback(project=args.wandb_project))

        checkpoint_config = CheckpointConfig(
            # num_to_keep=3,
            # checkpoint_score_attribute='sampler_results/episode_reward_mean',
            checkpoint_frequency=10,
        )

        tune.run(
            args.algo.upper(),
            name=config.env,
            stop={'timesteps_total': args.stop_timesteps},
            checkpoint_config=checkpoint_config,
            storage_path=f'{Path.cwd()}/ray_results/{config.env}',
            config=config.to_dict(),
            callbacks=callbacks
        )

        print('Finished training.')

    finally:
        print('Shutting down Ray.')
        ray.shutdown()


def get_checkpoint_paths(experiment_local_path: str) -> List[str]:
    """Get the paths to the checkpoints of an experiment."""
    return [str(p) for p in Path(experiment_local_path).glob('checkpoint_*')]


def extract_date_from_string(input_string: str) -> str:
    """
    Extracts a date formatted as 'YYYY-MM-DD-hh-mm-ss' from the end of a given
    string using regular expressions.

    Parameters:
    input_string (str): The input string from which the date is to
        be extracted.

    Returns:
    str or None: The extracted date in 'YYYY-MM-DD-hh-mm-ss' format if
    found, or None if not found.

    Example:
    >>> example_string = 'PPO_simple_speaker_listener_v4_4304a_00000_0_2023-09-07_11-32-44'
    >>> extract_date_from_string(example_string)
    '2023-09-07_11-32-44'
    """
    pattern = r'(\d{4}-\d{2}-\d{2}_\d+-\d+-\d+)'
    match = re.search(pattern, input_string)

    if match:
        return match.group(1)
    else:
        return None


def get_analysis(experiment_local_path: str,
                 default_mode='max',
                 default_metric='episode_reward_mean'
                 ) -> NewExperimentAnalysis:
    """
    Get the Ray analysis object for an experiment.
    """
    folders = experiment_local_path.split('/')
    if folders[-1] == '':
        folders = folders[:-1]

    experiments_dir = '/'.join(folders[:-1])
    timestamp = extract_date_from_string(folders[-1])

    try:
        return NewExperimentAnalysis(
            f'{experiments_dir}/experiment_state-{timestamp}.json',
            default_mode=default_mode,
            default_metric=default_metric)

    except FileNotFoundError:
        raise FileNotFoundError(
            f'Experiment analysis not found for {experiment_local_path}.')


def get_progress_df(experiment_local_path: str) -> pd.DataFrame:
    """Get the progress dataframe of an experiment."""
    progress_path = Path(experiment_local_path) / 'progress.csv'
    return pd.read_csv(progress_path)


def get_best_checkpoint(experiment_local_path: str,
                        default_mode='max',
                        default_metric='episode_reward_mean'
                        ) -> Checkpoint:
    """
    Get the best checkpoint of an experiment.
    """
    analysis = get_analysis(experiment_local_path,
                            default_mode=default_mode,
                            default_metric=default_metric)
    return analysis.best_checkpoint


def has_new_best_checkpoint(experiment_local_path: str,
                            last_known_best_checkpoint: Checkpoint,
                            default_mode='max',
                            default_metric='episode_reward_mean'
                            ) -> Checkpoint:
    """
    Check if an experiment has a new best checkpoint.
    """
    current_best_checkpoint = get_best_checkpoint(experiment_local_path,
                                                  default_mode=default_mode,
                                                  default_metric=default_metric)

    # compare the paths of the two checkpoints as Checkpoint objects do
    # not preserve identity with __eq__ or __hash__
    return current_best_checkpoint.path != last_known_best_checkpoint.path
