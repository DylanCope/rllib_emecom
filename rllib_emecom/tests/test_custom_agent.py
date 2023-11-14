from rllib_emecom.macrl.macrl_agent import MACRLAgent
from rllib_emecom.macrl.ppo.macrl_ppo import PPOMACRLConfig
from rllib_emecom.tests.test_train import get_macrl_ppo_module
from rllib_emecom.tests.utils import ray_init_wrapper


class CustomTestMACRLAgent(MACRLAgent):
    pass


@ray_init_wrapper
def test_custom_agent():
    config = (
        PPOMACRLConfig()
        .environment(
            'goal_comms_gridworld',
            env_config={'num_agents': 3},
            disable_env_checking=True,
            clip_actions=True
        )
        .communications(agent_ids=[f'agent_{i}' for i in range(3)])
        .agent_class(CustomTestMACRLAgent)
    )
    algo = config.build()
    ppo_macrl_module = get_macrl_ppo_module(algo)
    assert all(isinstance(agent, CustomTestMACRLAgent) for agent in ppo_macrl_module.agents.values()), \
        'Expected policy model to have CustomTestMACRLAgents'
