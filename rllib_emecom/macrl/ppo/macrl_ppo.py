from typing import Type, Union
from rllib_emecom.macrl.macrl_config import MACRLConfig
from rllib_emecom.macrl.macrl_module_spec import MACRLModuleSpec
from rllib_emecom.macrl.ppo.macrl_ppo_learner import PPOTorchMACRLLearner

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.core.learner.learner import Learner
# from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PartialAlgorithmConfigDict


class PPOMACRLConfig(PPOConfig, MACRLConfig):

    def __init__(self, algo_class=None):
        PPOConfig.__init__(self, algo_class=algo_class)
        MACRLConfig.__init__(self)

    @override(AlgorithmConfig)
    def to_dict(self) -> dict:
        return {**PPOConfig.to_dict(self), **MACRLConfig.to_dict(self)}

    @override(AlgorithmConfig)
    def update_from_dict(self, config_dict: PartialAlgorithmConfigDict) -> AlgorithmConfig:
        PPOConfig.update_from_dict(self, config_dict)
        MACRLConfig.update_from_dict(self, config_dict)
        return self

    @override(AlgorithmConfig)
    def validate(self):
        PPOConfig.validate(self)
        MACRLConfig.validate(self)

    @override(AlgorithmConfig)
    def get_default_rl_module_spec(self) -> MACRLModuleSpec:
        from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule

        return MACRLModuleSpec(
            comm_spec=self.create_comm_spec(),
            module_class=PPOTorchMACRLModule,
            catalog_class=PPOCatalog,
            model_config_dict=self.model
        )

    # @override(AlgorithmConfig)
    # def get_default_rl_module_spec(self) -> MACRLModuleSpec:
    #     from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule

    #     return SingleAgentRLModuleSpec(
    #         module_class=PPOTorchMACRLModule,
    #         catalog_class=PPOCatalog,
    #         model_config_dict={
    #             'comm_spec': self.create_comm_spec(),
    #             **self.model
    #         }
    #     )


class PPOMACRL(PPO):
    @classmethod
    @override(PPO)
    def get_default_config(cls) -> AlgorithmConfig:
        return PPOMACRLConfig()

    @override(AlgorithmConfig)
    def get_default_learner_class(self) -> Union[Type[Learner], str]:
        return PPOTorchMACRLLearner
