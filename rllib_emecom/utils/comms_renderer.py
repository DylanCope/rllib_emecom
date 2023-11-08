from rllib_emecom.macrl import AgentID
from rllib_emecom.utils.video_utils import register_video_wrapped_env, plot_to_array
from rllib_emecom.utils.video_callback import Renderer
from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule

from typing import Dict, Callable, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import networkx as nx

from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.evaluate import rollout
from ray.train import Checkpoint
from ray.rllib import Policy, BaseEnv
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID


Colour = Tuple[float, float, float]
MsgColourMap = Callable[[np.ndarray], Colour]

MessagesNetwork = Dict[AgentID, Dict[AgentID, np.ndarray]]
"""
A dictionary of messages sent between agents in the last forward pass
of the policy, in the form:
{
    receiver_id: {
        sender_id: msg
    }
}
"""


class OneHotMsgCircularColourMap:

    def __init__(self, n_msgs: int):
        self.n_msgs = n_msgs
        self.colour_map = sns.color_palette('hls', n_msgs)

    def __call__(self, msg: np.ndarray) -> Colour:
        assert msg.shape == (self.n_msgs, )
        msg_idx = np.argmax(msg)
        return self.colour_map[msg_idx]


class CommsRenderer(Renderer):

    def __init__(self, n_msgs: int, *args, **kwargs):
        self.msgs_colour_map = OneHotMsgCircularColourMap(n_msgs)
        self.agent_graph_pos = None
        sns.set()

    def render(
        self,
        env: BaseEnv,
        env_index: int,
        config: AlgorithmConfig,
        policies: Optional[Dict[PolicyID, Policy]],
        episode: Union[Episode, EpisodeV2],
        **kwargs,
    ) -> np.ndarray:
        frame = env.render()
        fig, axs = plt.subplots(1, 2, figsize=(6, 3))

        try:
            self._render_frame(frame, axs[0])
            msgs_network = self.get_msgs_network(policies, env_index)
            self._render_msgs_network(msgs_network, axs[1])
            fig.tight_layout()
            render = plot_to_array(fig)

        finally:
            plt.close(fig)

        return render

    def get_msgs_network(self,
                         policies: Optional[Dict[PolicyID, Policy]],
                         batch_index: int,
                         ) -> MessagesNetwork:
        """
        Constructs a dictionary of messages sent between agents in the last
        forward pass of the policy.

        The dictionary is of the form:
        {
            receiver_id: {
                sender_id: msg
            }
        }

        Args:
            policies: The 'policies' used to generate the messages.
                This dict should contain a single logical policy under the
                key 'default_policy' in order to allow message passing.
                This policy should use the PPOTorchMACRLModule.
            batch_index: The index of the batch to get the messages from.

        Returns:
            The messages network dictionary.
        """

        macrl_module = policies['default_policy'].model
        assert isinstance(macrl_module, PPOTorchMACRLModule)

        comm_spec = macrl_module.get_comms_spec()
        if macrl_module.last_msgs_sent is not None:
            return {
                receiver_id: {
                    sender_id: msgs_batch[batch_index, comm_spec.get_agent_idx(sender_id)]
                    for sender_id in comm_spec.comm_channels
                    if comm_spec.can_send(receiver_id, sender_id)
                }
                for receiver_id, msgs_batch in macrl_module.last_msgs_sent.items()
            }
        else:
            return None

    def _render_frame(self, frame, ax):
        ax.imshow(frame)
        ax.axis('off')

    def _render_msgs_network(self, msgs_network: MessagesNetwork, ax):
        agents = list(msgs_network.keys())
        agent_colours = dict(zip(agents, sns.color_palette(n_colors=len(agents))))

        G = nx.DiGraph()

        for receiver_id, senders in msgs_network.items():
            for sender_id, msg in senders.items():
                G.add_edge(receiver_id, sender_id,
                           colour=self.msgs_colour_map(msg))

        edge_colors = [data['colour'] for _, _, data in G.edges(data=True)]
        node_colors = [agent_colours[n] for n in G.nodes()]

        # Draw the graph with edge colors
        if self.agent_graph_pos is None:
            self.agent_graph_pos = nx.spring_layout(G, seed=42)
        nx.draw(G, self.agent_graph_pos,
                node_size=500,
                node_color=node_colors,
                edge_color=edge_colors,
                width=2,
                ax=ax,
                arrowsize=20,
                connectionstyle='arc3, rad = 0.1')


def create_video(algorithm: Algorithm,
                 episodes: int,
                 max_steps: int = 1000,
                 checkpoint: Optional[Checkpoint] = None,
                 output_video_path: Optional[str] = None):
    """
    Create a video from the best checkpoint.

    Args:
        experiment_local_path: The local path of the experiment.
        episodes: The number of episodes to run.
        max_steps: The maximum number of steps to run.

    Returns:
        The checkpoint used to make the video.
    """
    def create_renderer(policy, env):
        return CommsRenderer(env.render,
                             algorithm,
                             algorithm.config.env,
                             msgs_colour_map=None)

    video_env_id = register_video_wrapped_env(algorithm.config.env,
                                              output_video_path or 'videos',
                                              create_renderer)

    # Create a new algorithm with the video maker's video environment.
    new_config = algorithm.config.copy(copy_frozen=False)
    new_config.update_from_dict({'env': video_env_id})
    new_algo = new_config.build()

    if checkpoint is None:
        new_algo.restore(checkpoint)  # Restore the checkpoint for the new agent.

    rollout(new_algo, video_env_id,
            num_steps=max_steps, num_episodes=episodes)
