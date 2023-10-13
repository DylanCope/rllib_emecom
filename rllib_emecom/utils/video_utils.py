from rllib_emecom.utils.experiment_utils import get_best_checkpoint, initialise_ray
from rllib_emecom.utils.register_envs import get_registered_env_creator
from rllib_emecom.utils import get_best_grid

import ray
from ray.train import Checkpoint
from ray.rllib.evaluate import rollout
from ray.rllib.env import MultiAgentEnv
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env

from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import numpy as np
from IPython.display import HTML, clear_output
import base64
import imageio
import io

EnvRenderer = Callable[[], np.ndarray]


def embed_mp4(filename: str, clear_before=True, width=640, height=480) -> HTML:
    """Embeds an mp4 file in the notebook."""
    video = open(filename, 'rb').read()
    b64 = base64.b64encode(video)
    tag = '''
    <video width="{0}" height="{1}" controls>
    <source src="data:video/mp4;base64,{2}" type="video/mp4">
    Your browser does not support the video tag.
    </video>'''.format(width, height, b64.decode())

    if clear_before:
        clear_output()

    return HTML(tag)


def plot_to_array(fig) -> np.ndarray:
    """
    Converts a matplotlib figure to a numpy array.
    """
    io_buf = io.BytesIO()
    fig.savefig(io_buf, format='raw', facecolor='white', transparent=False)
    io_buf.seek(0)
    img_arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                         newshape=(int(fig.bbox.bounds[3]), int(fig.bbox.bounds[2]), -1))
    io_buf.close()
    return img_arr


def save_video(frames: List[np.ndarray],
               file_path: str = 'video',
               format='mp4',
               fps: int = 30) -> str:
    """
    Saves a video of the policy being evaluating in an environment.

    Args:
        frames (List[np.ndarray]): The frames of the video.
        file_path (str): The path of the file to save the video to.
        fps (int): The frames per second of the video.

    Returns:
        str: The path to the saved video.
    """
    assert len(frames) > 0, "No frames to save."
    if frames[0].dtype != np.uint8:
        frames = [frame.astype(np.uint8) for frame in frames]

    if not file_path.endswith(f'.{format}'):
        file_path = f'{file_path}.{format}'

    imageio.mimwrite(file_path, frames, fps=fps)

    return file_path


class VideoMakerStepWrapper:

    def __init__(self,
                 env,
                 output_dir: str,
                 fps: int = 5):
        self.env = env
        self.multiagent = isinstance(env, MultiAgentEnv)
        self.old_step = self.env.step
        self.frames = [[]]
        self.episodes = 0

        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.fps = fps

        env.step = self.step_wrapper

    def step_wrapper(self, *args, **kwargs):
        """ Wrapper for the render method of an environment. """
        step_results = self.old_step(*args, **kwargs)

        next_obs, reward, terminated, truncated, info = step_results
        if self.multiagent:
            terminated = terminated["__all__"]
            truncated = truncated["__all__"]

        frame = self.env.render()
        self.frames[self.episodes].append(frame)

        if terminated or truncated:
            self.create_video()
            self.episodes += 1
            self.frames.append([])

        return step_results

    def create_video(self):
        """ Create a video from the frames. """
        hash_id = str(hash(self))[:5]
        output_path = self.output_dir + f'/video_{hash_id}_{self.episodes + 1}'
        print(f'Writing video with {len(self.frames[self.episodes])} frames to: {output_path}')
        save_video(self.frames[self.episodes], output_path, fps=self.fps)


def register_video_wrapped_env(env_id: str,
                               output_dir: str,
                               create_renderer: Optional[Callable[[], EnvRenderer]] = None) -> str:
    """Register a video wrapped environment with Ray Tune and create a VideoMaker."""
    env_creator = get_registered_env_creator(env_id)

    def create_video_wrapped_env(config):
        env = env_creator(config)
        # renderer = create_renderer() if create_renderer else None
        VideoMakerStepWrapper(env, output_dir=output_dir)
        return env

    video_env_id = f'video_{env_id}'
    register_env(video_env_id, create_video_wrapped_env)

    return video_env_id


def create_best_checkpoint_video(experiment_local_path: str,
                                 episodes: int,
                                 max_steps: Optional[int] = None,
                                 output_video_path: Optional[str] = None) -> Checkpoint:
    """
    Create a video from the best checkpoint.

    Args:
        experiment_local_path: The local path of the experiment.
        episodes: The number of episodes to run.
        max_steps: The maximum number of steps to run.

    Returns:
        The checkpoint used to make the video.
    """
    max_steps = max_steps or 1000
    checkpoint = get_best_checkpoint(experiment_local_path)

    algo = Algorithm.from_checkpoint(checkpoint)
    video_env_id = register_video_wrapped_env(algo.config.env,
                                              f'{checkpoint.path}/videos')

    # Create a new algorithm with the video maker's video environment.
    new_config = algo.config.copy(copy_frozen=False)
    new_config.update_from_dict({'env': video_env_id})
    new_algo = new_config.build()
    new_algo.restore(checkpoint)  # Restore the checkpoint for the new agent.

    rollout(new_algo, video_env_id,
            num_steps=max_steps, num_episodes=episodes)

    return checkpoint


def create_merged_video_frames(frame_dict: Dict[Any, List[np.ndarray]],
                               w: int, h: int) -> List[np.ndarray]:
    """
    Create a list of frames for a video where each original video is played simultaneously
    in a (w, h) grid. Missing frames are filled with black.

    Parameters:
        frame_dict (Dict[int, List[np.ndarray]]): A dictionary mapping video IDs to lists of frames.
        w (int): The width of the grid.
        h (int): The height of the grid.

    Returns:
        List[np.ndarray]: A list of frames for the merged video.
    """
    # Get the dimensions of the base frames
    base_frame_shape = frame_dict[next(iter(frame_dict))][0].shape
    border = 2

    # Calculate the canvas dimensions based on the grid and frame dimensions
    canvas_width = base_frame_shape[0] * w + 2 * border * w
    canvas_height = base_frame_shape[1] * h + 2 * border * h
    canvas_depth = base_frame_shape[2]

    # Initialize an empty list to store frames for the video
    video_frames = []

    # Get the maximum number of frames from any video
    max_frames = max(len(frames) for frames in frame_dict.values())

    # Loop through each frame index
    for frame_index in range(max_frames):
        # Initialize an empty canvas for the current frame
        canvas = np.zeros((canvas_width, canvas_height, canvas_depth))

        for idx, video_id in enumerate(frame_dict):
            if frame_index < len(frame_dict[video_id]):
                frame = frame_dict[video_id][frame_index]
                x = idx % w
                y = idx // w
                x_start = x * base_frame_shape[0] + border * (x + 1)
                x_end = (x + 1) * base_frame_shape[0] + border * (x + 1)
                y_start = y * base_frame_shape[1] + border * (y + 1)
                y_end = (y + 1) * base_frame_shape[1] + border * (y + 1)

                # Place the frame on the canvas
                canvas[x_start:x_end, y_start:y_end, :] = frame

        # Add the canvas to the list of frames for the video
        video_frames.append(canvas)

    return video_frames


def create_grid_video(frame_dict: Dict[Any, List[np.ndarray]]) -> List[np.ndarray]:
    return create_merged_video_frames(frame_dict, *get_best_grid(len(frame_dict)))


if __name__ == '__main__':
    try:
        run_dir = '/home/ray/ray_results/PPO/PPO_simple_speaker_listener_v4_eab9f_00000_0_2023-09-07_17-13-52'
        initialise_ray()
        create_best_checkpoint_video(run_dir, 1)
    finally:
        print('Shutting down Ray.')
        ray.shutdown()
