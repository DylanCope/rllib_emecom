from rllib_emecom.macrl.macrl_config import MACRLConfig
from rllib_emecom.macrl.macrl_module_spec import MACRLModuleSpec

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import PartialAlgorithmConfigDict
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog


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
