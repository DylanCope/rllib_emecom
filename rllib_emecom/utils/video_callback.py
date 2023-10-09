from zmq import has
from rllib_emecom.utils.general import get_timestamp
from rllib_emecom.utils.video_utils import create_grid_video, save_video

from typing import Any, List, Optional, Dict, Union
from pathlib import Path
from wandb import Video
import numpy as np

from ray.rllib import Policy, BaseEnv
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.vector_env import VectorEnvWrapper
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.evaluation.rollout_worker import RolloutWorker


def get_sub_env(base_env, idx):
    if isinstance(base_env, VectorEnvWrapper):
        sub_envs = base_env.vector_env.get_sub_environments()
    else:
        sub_envs = base_env.envs
    return sub_envs[idx]


class Renderer:

    def render(
        self,
        env: BaseEnv,
        env_index: int,
        config: AlgorithmConfig,
        episode: Union[Episode, EpisodeV2],
        policies: Optional[Dict[PolicyID, Policy]],
        **kwargs,
    ) -> np.ndarray:
        return env.render()


class VideoMakingManager:

    def __init__(self, renderer: Renderer):
        self.frames = []
        self.finished = False
        self.renderer = renderer

    def is_finished(self):
        return self.finished

    def render(self, *args, **kwargs):
        frame = self.renderer.render(*args, **kwargs)
        self.frames.append(frame)


class VideoEvaluationsCallback(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.episodes_executed = 0
        self.max_videos = 16
        self.video_managers: Dict[int, VideoMakingManager] = {}
        self.video_saved = False

    @property
    def videos_created(self):
        return sum(
            manager.is_finished()
            for manager in self.video_managers.values()
        )

    def frames_key(self, episode: Union[Episode, EpisodeV2]) -> int:
        return episode.episode_id

    def _get_videos_dir(self, worker: RolloutWorker):
        videos_dir = f'{worker.io_context.log_dir}/videos/'
        Path(videos_dir).mkdir(parents=True, exist_ok=True)
        return videos_dir

    def on_episode_end(
            self,
            *,
            worker: RolloutWorker,
            base_env: BaseEnv,
            policies: Dict[PolicyID, Policy],
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None,
            **kwargs,
    ) -> None:
        if worker.config["in_evaluation"]:
            video_manager = self.get_video_manager(worker, episode)
            video_manager.finished = True
            self.on_eval_episode_end(worker, episode, env_index)

    def on_eval_episode_end(
            self,
            worker: RolloutWorker,
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None):

        if self.videos_created >= self.max_videos and not self.video_saved:
            frames = create_grid_video(self.get_frames_dict())

            videos_dir = self._get_videos_dir(worker)
            video_file = f'{videos_dir}/video_{get_timestamp()}.mp4'
            path = save_video(frames, file_path=video_file, fps=5)
            print(f"Saved evaluation video to {path}")
            self.video_saved = True

            episode.media[f"env_{env_index}_video"] = Video(path)

            self.reset()

    def get_frames_dict(self) -> Dict[int, List[np.ndarray]]:
        return {
            key: manager.frames
            for key, manager in self.video_managers.items()
        }

    def get_video_manager(self,
                          worker: RolloutWorker,
                          episode: Union[Episode, EpisodeV2]) -> VideoMakingManager:
        managers_key = self.frames_key(episode)
        if managers_key not in self.video_managers:
            renderer_cls = worker.config['env_config'].get('renderer_cls', Renderer)
            renderer_conf = worker.config['env_config'].get('renderer_config', {})
            renderer = renderer_cls(**renderer_conf)
            self.video_managers[managers_key] = VideoMakingManager(renderer)
        return self.video_managers[managers_key]

    def on_episode_step(
        self,
        *,
        worker,
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ) -> None:

        if worker.config["in_evaluation"] and self.videos_created < self.max_videos:
            video_manager = self.get_video_manager(worker, episode)
            env = get_sub_env(base_env, env_index)
            video_manager.render(env, env_index, worker.config, policies, episode)

            # info = {
            #     agent: {
            #         'reward': episode.agent_rewards[agent],
            #         **episode.last_info_for(agent)
            #     }
            #     for (agent, _) in episode.agent_rewards
            # }

            # frames.append((frame, info))
