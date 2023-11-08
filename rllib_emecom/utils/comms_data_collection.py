from typing import Optional, Dict, Union

import numpy as np
import pandas as pd

from ray.rllib.core.models.base import SampleBatch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import PolicyID
from ray.rllib import Policy, BaseEnv

from rllib_emecom.utils.video_callback import get_sub_env
from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule
from rllib_emecom.env.goal_comms_gridworld import parallel_env as GoalCommsGridworldEnv


class CollectCommsDataCallback(DefaultCallbacks):

    def __init__(self):
        super().__init__()
        self.data = []

    def get_dataframe(self) -> pd.DataFrame:

        def get_goal_vec(item, agent_id) -> np.array:
            goals = item['goals']
            goal = item['env_state']['goal_positions'][agent_id]
            return np.array([(g == goal).all() for g in goals], np.int32)

        def get_msg_cols(item, recv_agent, send_agent):
            return {
                f'msg_{i}': v
                for i, v in enumerate(item['comms'][recv_agent][send_agent])
            }

        df = pd.DataFrame([
            {
                'receiver': recv_agent,
                'sender': send_agent,
                'goal_x': item['env_state']['goal_positions'][recv_agent][0],
                'goal_y': item['env_state']['goal_positions'][recv_agent][1],
                'pos_x': item['env_state']['agent_positions'][recv_agent][0],
                'pos_y': item['env_state']['agent_positions'][recv_agent][1],
                'sender_pos_x': item['env_state']['agent_positions'][send_agent][0],
                'sender_pos_y': item['env_state']['agent_positions'][send_agent][1],
                'world_w': item['world_w'],
                'world_h': item['world_h'],
                'receiver_action': np.argmax(item['last_action_logits'][recv_agent]),
                'msg': np.argmax(item['comms'][recv_agent][send_agent]),
                'goal_vec': get_goal_vec(item, recv_agent),
                **get_msg_cols(item, recv_agent, send_agent),
            }
            for item in self.data
            for recv_agent in item['comms']
            for send_agent in item['comms'][recv_agent]
            if item['env_state']['goal_knowledge_map'][send_agent] == recv_agent
        ])
        df['goal_grid_idx'] = df['goal_x'] + df['goal_y'] * df['world_w']
        df['pos_grid_idx'] = df['pos_x'] + df['pos_y'] * df['world_w']
        return df

    def get_env_state(self, goal_comms_env: GoalCommsGridworldEnv) -> dict:
        agent_ids = goal_comms_env.agents

        def get_known_goal_agent(agent):
            # agent knows the goal of agent_goal_map[agent]
            other_agent = goal_comms_env.agent_goal_map[agent]
            return agent_ids[other_agent.index]

        return {
            # key knows the goal of the value
            # so key is the sender and value is the receiver
            'goal_knowledge_map' : {
                agent: get_known_goal_agent(agent)
                for agent in agent_ids
            },
            'agent_positions': {
                agent: goal_comms_env.agents_map[agent].pose
                for agent in agent_ids
            },
            'goal_positions': {
                agent: goal_comms_env.agents_map[agent].goal
                for agent in agent_ids
            },
        }

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
        macrl_module = policies['default_policy'].model
        assert isinstance(macrl_module, PPOTorchMACRLModule)
        env = get_sub_env(base_env, env_index)
        goal_comms_env = env.env.unwrapped.par_env
        assert isinstance(goal_comms_env, GoalCommsGridworldEnv)

        last_action_logits = {
            agent_id: action_batch[SampleBatch.ACTION_DIST_INPUTS][env_index].detach().cpu().numpy()
            for agent_id, action_batch in macrl_module.last_actor_outputs.items()
        }

        self.data.append({
            'comms': macrl_module.get_last_msgs(env_index),
            'total_env_steps': episode.total_env_steps,
            'total_reward': episode.total_reward,
            'episode_id': episode.episode_id,
            'env_index': env_index,
            'last_obs': macrl_module.last_inputs[SampleBatch.OBS][env_index],
            'last_action_logits': last_action_logits,
            'env_state': self.get_env_state(goal_comms_env),
            'goals': np.array(goal_comms_env.goals),
            'world_w': goal_comms_env.world_shape[0],
            'world_h': goal_comms_env.world_shape[1],
        })
