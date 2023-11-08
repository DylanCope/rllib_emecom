# Environment adapted from: https://github.com/proroklab/rllib_differentiable_comms

import itertools
import os
import functools
from typing import Optional, Tuple
import seaborn as sns

import pygame
from pygame import gfxdraw
import numpy as np
import gymnasium
from gymnasium.spaces import Discrete, Box

from pettingzoo import ParallelEnv
from pettingzoo.utils import parallel_to_aec, wrappers


CELL_SIZE = 32
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
TIME_COST = 1

X = 1
Y = 0


def draw_circle(surface, color, x, y, radius):
    gfxdraw.aacircle(surface, x, y, radius, color)
    gfxdraw.filled_circle(surface, x, y, radius, color)


def get_sns_colour_palette(n_colours: int, pallete: Optional[str] = None) -> list:
    return [
        (int(255 * r), int(255 * g), int(255 * b))
        for (r, g, b) in sns.color_palette(pallete, n_colours)
    ]


def pastel_color_generator(n_colours: int) -> tuple:
    return itertools.cycle(get_sns_colour_palette(n_colours, 'pastel'))


def bright_color_generator(n_colours: int) -> tuple:
    return itertools.cycle(get_sns_colour_palette(n_colours))


def one_hot(x: int, n: int) -> np.ndarray:
    v = np.zeros(n)
    v[x] = 1
    return v.astype(np.float32)


class BaseAgent:

    def __init__(self, index, world_shape, random_state):
        self.goal = None
        self.pose = None
        self.reached_goal = None
        self.known_goal = None
        self.random_state = random_state
        self.index = index
        self.world_shape = world_shape
        self.reset()

    def is_valid_pose(self, p):
        return all([0 <= p[c] < self.world_shape[c] for c in [Y, X]])

    def update_pose(self, delta_p):
        desired_pos = self.pose + delta_p
        if self.is_valid_pose(desired_pos):
            self.pose = desired_pos

    def reset(self):
        raise NotImplementedError()

    def step(self, action):
        raise NotImplementedError()


class DiscreteAgent(BaseAgent):

    ACTION_STRINGS = ["stay", "down", "up", "left", "right"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset(self):
        self.pose = self.random_state.randint((0, 0), self.world_shape)
        self.goal = self.random_state.randint((0, 0), self.world_shape)
        self.reached_goal = False
        return 0

    def step(self, action):
        delta_pose = {
            0: [0, 0],
            1: [0, 1],
            2: [0, -1],
            3: [-1, 0],
            4: [1, 0],
        }[action]
        self.update_pose(delta_pose)


def env(render_mode=None, **config):
    """
    The env function often wraps the environment in wrappers by default.
    You can find full documentation for these methods
    elsewhere in the developer documentation.
    """
    internal_render_mode = render_mode if render_mode != "ansi" else "human"
    env = raw_env(render_mode=internal_render_mode, **config)
    # This wrapper is only for environments which print results to the terminal
    if render_mode == "ansi":
        env = wrappers.CaptureStdoutWrapper(env)
    # this wrapper helps error handling for discrete action spaces
    env = wrappers.AssertOutOfBoundsWrapper(env)
    # Provides a wide vareity of helpful user errors
    # Strongly recommended
    env = wrappers.OrderEnforcingWrapper(env)
    return env


def raw_env(**config):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(**config)
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self,
                 world_shape: Tuple[int, int] = (5, 5),
                 num_agents: int = 3,
                 goal_shift: int = 1,
                 max_episode_len: int = 10,
                 random_state: np.random.RandomState = None,
                 scalar_obs: bool = False,
                 render_mode: str = 'rgb_array',
                 observe_others_pos: bool = False,
                 observe_goals: bool = True,
                 observe_self_id: bool = True,
                 **kwargs):
        """
        The init method takes in environment arguments and should define the following attributes:
        - possible_agents
        - render_mode

        Note: as of v1.18.1, the action_spaces and observation_spaces attributes are deprecated.
        Spaces should be defined in the action_space() and observation_space() methods.
        If these methods are not overridden, spaces will be inferred from
        self.observation_spaces/action_spaces, raising a warning.

        These attributes should not be changed after initialization.
        """
        self.world_shape = world_shape
        self.random_state = random_state or np.random.RandomState()
        self.max_episode_len = max_episode_len
        self.use_scalar_obs = scalar_obs
        self.observe_others_pos = observe_others_pos
        self.observe_self_id = observe_self_id
        self.observe_goals = observe_goals

        self.agents_map = {
            f"agent_{i}": DiscreteAgent(i, self.world_shape, self.random_state)
            for i in range(num_agents)
        }
        self.possible_agents = list(self.agents_map.keys())
        self.agents = list(self.agents_map.keys())
        self.n_agents = len(self.agents)

        self.agent_goal_map = {
            agent_id: self.agents_map[self.agents[(idx + goal_shift) % num_agents]]
            for idx, agent_id in enumerate(self.possible_agents)
        }

        self.goals = None

        # # optional: a mapping between agent name and ID
        # self.agent_name_mapping = dict(
        #     zip(self.possible_agents, list(range(len(self.possible_agents))))
        # )
        self.render_mode = render_mode
        if self.render_mode == 'rgb_array':
            os.environ["SDL_VIDEODRIVER"] = "dummy"

        pygame.display.init()
        self.pygame_window = None

    # Observation space should be defined here.
    # lru_cache allows observation and action spaces to be memoized,
    # reducing clock cycles required to get each agent's space.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # gymnasium spaces are defined and documented here: https://gymnasium.farama.org/api/spaces/
        if self.use_scalar_obs:
            assert not self.observe_others_pos, \
                "Scalar obs not currently supported with observe_others_pos"
            return Box(
                low=np.array([0, 0, 0, 0]),
                high=np.array([*self.world_shape, *self.world_shape]),
                shape=(4,)
            )

        # n_pos_obs = self.n_agents if self.observe_others_pos else 1
        # obs_dim = self.world_shape[X] * (1 + n_pos_obs) \
        #     + self.world_shape[Y] * (1 + n_pos_obs)

        # if self.observe_self_id:
        #     obs_dim += self.n_agents

        # if self.observe_goals:
        #     obs_dim += self.n_agents * 2
        agent, *_ = self.agents_map.values()
        obs = self.get_agent_obs(agent)
        return Box(low=0, high=1, shape=obs.shape)

    # Action space should be defined here.
    # If your spaces change over time, remove this line (disable caching).
    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return Discrete(5)

    def to_string(self) -> str:
        top_bot_margin = " " + "-" * self.world_shape[Y] * 2 + "\n"
        r = top_bot_margin
        for y in range(self.world_shape[Y]):
            r += "|"
            for x in range(self.world_shape[X]):
                c = " "
                for i, agent in enumerate(self.agents_map.values()):
                    if np.all(agent.pose.astype(int) == np.array([y, x])):
                        c = "x" if agent.reached_goal else str(i)
                    if np.all(agent.goal == np.array([y, x])):
                        c = "abcdefghijklmnopqrstuvwxyz"[i]
                r += c + " "
            r += "|\n"
        r += top_bot_margin
        return r

    def render_pygame(self) -> np.ndarray:
        world_width = self.world_shape[X]
        world_height = self.world_shape[Y]

        if self.pygame_window is None:
            window_width = int(CELL_SIZE * world_width)
            window_height = int(CELL_SIZE * world_height)
            self.pygame_window = pygame.display.set_mode((window_width, window_height))

        self.pygame_window.fill(WHITE)

        def get_cell_rect(x: int, y: int):
            return (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE)

        def get_cell_circle(x: int, y: int):
            centre_x = y * CELL_SIZE + CELL_SIZE // 2
            centre_y = x * CELL_SIZE + CELL_SIZE // 2
            r = int(0.8 * CELL_SIZE // 2)
            return (centre_x, centre_y, r)

        for agent, colour in zip(self.agents_map.values(),
                                 pastel_color_generator(self.n_agents)):
            pygame.draw.rect(self.pygame_window, colour, get_cell_rect(*agent.goal))

        for agent, colour in zip(self.agents_map.values(),
                                 bright_color_generator(self.n_agents)):
            draw_circle(self.pygame_window, colour, *get_cell_circle(*agent.pose))

        return pygame.surfarray.array3d(pygame.display.get_surface())

    def render(self):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.render_mode == 'rgb_array':
            return self.render_pygame()

        # if self.render_mode == 'human':
        #     self.render_pygame()
        #     pygame.display.update()

        print(self.to_string())

    def close(self):
        """
        Close should release any graphical displays, subprocesses, network connections
        or any other environment data which should not be kept around after the
        user is no longer using the environment.
        """
        pygame.quit()

    def reset(self, seed=None, options=None):
        """
        Reset needs to initialize the `agents` attribute and must set up the
        environment so that render(), and step() can be called without issues.
        Here it initializes the `num_moves` variable which counts the number of
        hands that are played.
        Returns the observations for each agent
        """
        # self.agents = self.possible_agents[:]
        # self.num_moves = 0
        # observations = {agent: NONE for agent in self.agents}
        # infos = {agent: {} for agent in self.agents}
        # self.state = observations
        # reset_actions = [agent.reset() for agent in self.agents]
        for agent in self.agents_map.values():
            # random new pos and goal
            agent.reset()

        for agent_id, agent in self.agents_map.items():
            agent.known_goal = self.agent_goal_map[agent_id].goal

        self.timestep = 0

        self.goals = [
            agent.goal for agent in self.agents_map.values()
        ]
        np.random.shuffle(self.goals)

        return self.get_observations(), self.get_infos()

    def get_agent_obs(self, agent: BaseAgent):
        if agent.known_goal is None:
            raise ValueError("known_goal is None, cannot construction observation.")

        feat_vecs = []
        if not self.use_scalar_obs:
            world_w, world_h = self.world_shape
            goal_x_1h = one_hot(agent.known_goal[X], world_w)
            goal_y_1h = one_hot(agent.known_goal[Y], world_h)
            feat_vecs += [goal_x_1h, goal_y_1h]

            if self.observe_others_pos:
                feat_vecs += [one_hot(a.pose[X], world_w)
                              for a in self.agents_map.values()]
                feat_vecs += [one_hot(a.pose[Y], world_h)
                              for a in self.agents_map.values()]

            else:
                feat_vecs += [one_hot(agent.pose[X], world_w),
                              one_hot(agent.pose[Y], world_h)]
        else:
            feat_vecs += [agent.known_goal, agent.pose]

        if self.observe_self_id:
            feat_vecs.append(one_hot(agent.index, self.n_agents))

        if self.observe_goals:
            for goal_x, goal_y in self.goals:
                feat_vecs.append(one_hot(goal_x, self.world_shape[X]))
                feat_vecs.append(one_hot(goal_y, self.world_shape[Y]))

        return np.hstack(feat_vecs).astype(np.float32)

    def get_observations(self):
        return {
            agent_id: self.get_agent_obs(agent)
            for agent_id, agent in self.agents_map.items()
        }

    def get_infos(self):
        return {agent: {} for agent in self.agents}

    def step(self, actions: dict):
        """
        step(action) takes in an action for each agent and should return the
        - observations
        - rewards
        - terminations
        - truncations
        - infos
        dicts where each dict looks like {agent_1: item_1, agent_2: item_2}
        """
        # If a user passes in actions with no agents, then just return empty observations, etc.
        if not actions:
            self.agents = []
            return {}, {}, {}, {}, {}

        rewards = {}

        all_finished = True
        for agent_id, action in actions.items():
            agent = self.agents_map[agent_id]
            agent.step(action)
            rewards[agent_id] = -TIME_COST if not agent.reached_goal else 0
            if not agent.reached_goal and all(agent.pose == agent.goal):
                rewards[agent_id] = 1
                agent.reached_goal = True
            all_finished = all_finished and agent.reached_goal

        terminations = {
            agent_id: all_finished
            for agent_id in self.agents_map.keys()
        }

        truncate = self.timestep >= self.max_episode_len
        truncations = {
            agent_id: truncate
            for agent_id in self.agents_map.keys()
        }

        self.timestep += 1

        return self.get_observations(), rewards, terminations, truncations, self.get_infos()


GoalCommsGridworldEnv = parallel_env
