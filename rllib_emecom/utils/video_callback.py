from rllib_emecom.utils.general import get_timestamp
from rllib_emecom.utils.video_utils import create_grid_video, save_video

from typing import Any, List, Optional, Dict, Union
from pathlib import Path
from wandb import Video
import numpy as np

from ray.rllib import Policy, BaseEnv
from ray.rllib.algorithms.algorithm import Algorithm
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
        policies: Optional[Dict[PolicyID, Policy]],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> np.ndarray:
        return env.render()


class VideoMakingManager:

    def __init__(self, renderer: Renderer):
        self.frames = []
        self.finished = False
        self.renderer = renderer

    def is_finished(self):
        return self.finished and len(self.frames) > 0

    def render(self, *args, **kwargs):
        frame = self.renderer.render(*args, **kwargs)
        self.frames.append(frame)


class VideoEvaluationsCallback(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset()

    def reset(self):
        self.video_managers: Dict[int, VideoMakingManager] = {}
        self.video_saved = False

    @property
    def n_episodes_completed(self):
        return sum(
            manager.is_finished()
            for manager in self.video_managers.values()
        )

    def get_render_config(self, config: AlgorithmConfig) -> Dict[str, Any]:
        return config['env_config'].get('render_config', {})

    def get_fps(self, config: AlgorithmConfig) -> int:
        render_conf = self.get_render_config(config)
        return render_conf.get('fps', 5)

    def get_episodes_per_video(self, config: AlgorithmConfig) -> int:
        render_conf = self.get_render_config(config)
        return render_conf.get('episodes_per_video', 8)

    def frames_key(self, episode: Union[Episode, EpisodeV2]) -> int:
        return episode.episode_id

    def _get_videos_dir(self, worker: RolloutWorker):
        videos_dir = f'{worker.io_context.log_dir}/videos/'
        Path(videos_dir).mkdir(parents=True, exist_ok=True)
        return videos_dir

    def on_evaluate_start(
        self, *, algorithm: Algorithm, **kwargs,
    ) -> None:
        self.reset()

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
            video_manager = self.get_video_manager(worker.config, episode)
            video_manager.finished = True
            self.on_eval_episode_end(worker, episode, env_index)

    def ready_to_save_video(self, config: AlgorithmConfig) -> bool:
        return self.n_episodes_completed >= self.get_episodes_per_video(config)

    def on_eval_episode_end(
            self,
            worker: RolloutWorker,
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None):

        if self.ready_to_save_video(worker.config) and not self.video_saved:
            frames = create_grid_video(self.get_frames_dict())

            videos_dir = self._get_videos_dir(worker)
            video_file = f'{videos_dir}/video_{get_timestamp()}.mp4'
            path = save_video(frames,
                              file_path=video_file,
                              fps=self.get_fps(worker.config))
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
                          config: AlgorithmConfig,
                          episode: Union[Episode, EpisodeV2]) -> VideoMakingManager:
        managers_key = self.frames_key(episode)
        if managers_key not in self.video_managers:
            render_conf = self.get_render_config(config)
            renderer_cls = render_conf.get('renderer_cls', Renderer)
            renderer = renderer_cls(**render_conf)
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

        if worker.config["in_evaluation"] and not self.video_saved:
            video_manager = self.get_video_manager(worker.config, episode)
            env = get_sub_env(base_env, env_index)
            try:
                video_manager.render(env, env_index, worker.config, policies, episode)
            except Exception:
                pass
