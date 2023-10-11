from rllib_emecom.macrl.ppo.macrl_ppo_module import PPOTorchMACRLModule

from ray.rllib.core.learner.learner import Learner
from ray.rllib.algorithms.ppo.torch.ppo_torch_learner import PPOTorchLearner
from ray.rllib.utils.annotations import override


class PPOTorchMACRLLearner(PPOTorchLearner):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iteration = 0

    @override(Learner)
    def update(self, *args, **kwargs):
        results = super().update(*args, **kwargs)
        self.iteration += 1
        self.update_hyperparameters()
        return results

    def update_hyperparameters(self):
        for module_id, module in self.module._rl_modules.items():
            if isinstance(module, PPOTorchMACRLModule):
                hyperparams = module.update_hyperparameters(self.iteration)
                self.register_metrics(module_id, hyperparams)
