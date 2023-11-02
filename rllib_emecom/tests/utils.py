from rllib_emecom.utils.experiment_utils import initialise_ray

from argparse import ArgumentParser, Namespace
import ray


def ray_init_wrapper(func):
    """Decorator for functions that use Ray."""

    def wrapper(*args, **kwargs):
        try:
            initialise_ray()
            return func(*args, **kwargs)
        finally:
            ray.shutdown()

    return wrapper


def parse_default_args(parser: ArgumentParser,
                       **override_args) -> Namespace:
    args, _ = parser.parse_known_args([
        item
        for k, v in override_args.items()
        for item in (
            [f'--{k}', str(v)]
            if not isinstance(v, bool)
            else [f'--{k}']
        )
    ])
    return args
