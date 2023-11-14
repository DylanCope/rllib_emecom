from typing import Type, Union, Optional
from rllib_emecom.macrl.macrl_config import MACRLConfig

from ray.rllib.algorithms.dqn import DQN, DQNConfig
from ray.rllib.algorithms.simple_q.simple_q import SimpleQ
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.learner.learner import Learner
# from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.rllib.policy.policy import Policy


class DQNMACRLConfig(DQNConfig, MACRLConfig):

    def __init__(self, algo_class=None):
        DQNConfig.__init__(self, algo_class=algo_class)
        MACRLConfig.__init__(self)

    @override(AlgorithmConfig)
    def to_dict(self) -> dict:
        return {**DQNConfig.to_dict(self), **MACRLConfig.to_dict(self)}

    @override(AlgorithmConfig)
    def update_from_dict(self, config_dict: PartialAlgorithmConfigDict) -> AlgorithmConfig:
        DQNConfig.update_from_dict(self, config_dict)
        MACRLConfig.update_from_dict(self, config_dict)
        return self

    @override(AlgorithmConfig)
    def validate(self):
        DQNConfig.validate(self)
        MACRLConfig.validate(self)

    @classmethod
    @override(SimpleQ)
    def get_default_policy_class(
        cls, config: AlgorithmConfig
    ) -> Optional[Type[Policy]]:
        if config["framework"] == "torch":
            return DQNTorchPolicy
        else:
            return DQNTFPolicy

    # @override(AlgorithmConfig)
    # def get_default_rl_module_spec(self) -> MACRLModuleSpec:
    #     from rllib_emecom.macrl.dqn.macrl_dqn_module import DQNTorchMACRLModule

    #     return MACRLModuleSpec(
    #         comm_spec=self.create_comm_spec(),
    #         module_class=DQNTorchMACRLModule,
    #         catalog_class=DQNCatalog,
    #         model_config_dict=self.model
    #     )

    # @override(AlgorithmConfig)
    # def get_default_rl_module_spec(self) -> MACRLModuleSpec:
    #     from rllib_emecom.macrl.dqn.macrl_dqn_module import DQNTorchMACRLModule

    #     return SingleAgentRLModuleSpec(
    #         module_class=DQNTorchMACRLModule,
    #         catalog_class=DQNCatalog,
    #         model_config_dict={
    #             'comm_spec': self.create_comm_spec(),
    #             **self.model
    #         }
    #     )


class DQNMACRL(DQN):
    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfig:
        return DQNMACRLConfig()

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type[Learner], str]:
        return DQNTorchMACRLLearner
