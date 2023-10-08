from rllib_emecom.utils.general import get_timestamp
from rllib_emecom.utils.video_utils import create_grid_video, save_video

from typing import Any, Optional, Dict, Union
from pathlib import Path
from wandb import Video

from ray.rllib import Policy, BaseEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.vector_env import VectorEnvWrapper
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib.evaluation.rollout_worker import RolloutWorker


class VideoEvaluationsCallback(DefaultCallbacks):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.episodes_executed = 0
        self.max_videos = 16
        self.reset()

    def reset(self):
        self.frames_dict = {}
        self.video_saved = False
        self.videos_finished = []

    def on_evaluate_start(
        self, *, algorithm: "Algorithm",
        **kwargs,
    ) -> None:
        """Callback before evaluation starts.

        This method gets called at the beginning of Algorithm.evaluate().

        Args:
            algorithm: Reference to the algorithm instance.
            kwargs: Forward compatibility placeholder.
        """
        self.reset()

    @property
    def videos_created(self):
        return len(self.videos_finished)

    def _get_envs(self, base_env):
        if isinstance(base_env, VectorEnvWrapper):
            return base_env.vector_env.get_sub_environments()
        else:
            return base_env.envs

    def frames_key(self, episode: Union[Episode, EpisodeV2]) -> Any:
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
        frames_key = self.frames_key(episode)
        if frames_key in self.frames_dict and frames_key not in self.videos_finished:
            self.videos_finished.append(frames_key)

        if worker.config["in_evaluation"]:
            self.on_eval_episode_end(worker, episode, env_index)

    def on_eval_episode_end(
            self,
            worker: RolloutWorker,
            episode: Union[Episode, EpisodeV2, Exception],
            env_index: Optional[int] = None):

        if self.videos_created >= self.max_videos and not self.video_saved:
            frames = create_grid_video(self.frames_dict)

            videos_dir = self._get_videos_dir(worker)
            video_file = f'{videos_dir}/video_{get_timestamp()}.mp4'
            path = save_video(frames, file_path=video_file, fps=5)
            print(f"Saved evaluation video to {path}")
            self.video_saved = True

            episode.media[f"env_{env_index}_video"] = Video(path)

            self.reset()

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
            frames_key = self.frames_key(episode)

            frames = self.frames_dict.get(frames_key, [])
            frame = self._get_envs(base_env)[env_index].render()

            # info = {
            #     agent: {
            #         'reward': episode.agent_rewards[agent],
            #         **episode.last_info_for(agent)
            #     }
            #     for (agent, _) in episode.agent_rewards
            # }

            # frames.append((frame, info))

            frames.append(frame)

            self.frames_dict[frames_key] = frames
