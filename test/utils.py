from rllib_emecom.utils.experiment_utils import initialise_ray

import ray


def ray_wrapper(func):
    """Decorator for functions that use Ray."""

    def wrapper(*args, **kwargs):
        try:
            initialise_ray()
            return func(*args, **kwargs)
        finally:
            ray.shutdown()

    return wrapper