from rllib_emecom.env import goal_comms_gridworld

from pettingzoo.mpe import (
    simple_speaker_listener_v4, simple_reference_v3,
    simple_v3, simple_spread_v3
)
from pettingzoo.butterfly import pistonball_v6
from gymnasium.spaces import Tuple
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.tune.registry import register_env, ENV_CREATOR, _global_registry
import supersuit as ss


def patched_action_space_sampler(self, agents):
    return {
        agent: self.par_env.action_space(agent).sample()
        for agent in self.get_agent_ids()
        if agent in agents
    }


def create_env_maker(create_pettingzoo_env, wrappers=None):
    wrappers = wrappers or []

    def create_env(config) -> ParallelPettingZooEnv:
        env = create_pettingzoo_env(**config)

        for wrapper in wrappers:
            env = wrapper(env)

        ray_env = ParallelPettingZooEnv(env)
        ray_env._agent_ids = env.agents
        ray_env.reset()
        # ray_env = ray_env.with_agent_groups({'agents': ray_env._agent_ids})

        # Patching issues with converting the env to a ray env.
        # Patch the action space sampler to sample from the parallel env.
        ray_env.action_space_sample = \
            lambda agents: patched_action_space_sampler(ray_env, agents)

        # Patch the render function to render from the parallel env.
        ray_env.render = lambda: ray_env.par_env.render()

        grouped_observation_space = Tuple([ray_env.observation_space] * len(ray_env._agent_ids))
        grouped_action_space = Tuple([ray_env.action_space] * len(ray_env._agent_ids))

        grouped_env = ray_env.with_agent_groups(
            {'agents': ray_env._agent_ids},
            obs_space=grouped_observation_space,
            act_space=grouped_action_space
        )
        grouped_env.render = lambda: ray_env.par_env.render()

        return grouped_env

    return create_env


def register_envs():
    modules = [
        (simple_speaker_listener_v4, [ss.pad_observations_v0, ss.pad_action_space_v0]),
        (simple_reference_v3, [ss.pad_observations_v0, ss.pad_action_space_v0]),
        # (silent_simple_reference, [ss.pad_observations_v0, ss.pad_action_space_v0]),
        (simple_spread_v3, [ss.pad_observations_v0, ss.pad_action_space_v0]),
        (simple_v3, []),
        (pistonball_v6, [
            lambda env: ss.color_reduction_v0(env, mode="B"),
            lambda env: ss.dtype_v0(env, "float32"),
            lambda env: ss.resize_v1(env, x_size=84, y_size=84),
            lambda env: ss.normalize_obs_v0(env, env_min=0, env_max=1),
            lambda env: ss.frame_stack_v1(env, 3),
        ]),
        (goal_comms_gridworld, [])
    ]

    for module, wrappers in modules:
        create_env = create_env_maker(module.parallel_env, wrappers)
        # ray.rllib.utils.check_env(create_env({}))
        register_env(module.__name__.split('.')[-1], create_env)


def get_registered_env_creator(env_id: str) -> callable:
    if _global_registry.contains(ENV_CREATOR, env_id):
        return _global_registry.get(ENV_CREATOR, env_id)

    raise ValueError(f'Environment {env_id} not registered.')
