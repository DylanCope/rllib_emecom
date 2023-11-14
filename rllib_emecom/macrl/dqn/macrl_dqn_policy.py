
from ray.rllib.algorithms.dqn.dqn_torch_policy import (
    build_q_losses,
    get_distribution_inputs_and_class,
    build_q_stats,
    postprocess_nstep_and_prio,
    adam_optimizer,
    grad_process_and_td_error_fn,
    concat_multi_gpu_td_errors,
    extra_action_out_fn,
    setup_early_mixins,
    before_loss_init,
    ComputeTDErrorMixin,
)
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.policy.torch_mixins import (
    LearningRateSchedule,
    TargetNetworkMixin,
)
from ray.rllib.models.torch.torch_action_dist import (
    get_torch_categorical_class_with_temperature,
    TorchDistributionWrapper,
)
from ray.rllib.algorithms.dqn.dqn_tf_policy import (
    Q_SCOPE,
    Q_TARGET_SCOPE,
    postprocess_nstep_and_prio,
)
from ray.rllib.policy.policy import Policy
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.typing import AlgorithmConfigDict
from ray.rllib.utils.error import UnsupportedSpaceException
from ray.rllib.utils.exploration.parameter_noise import ParameterNoise

import gymnasium as gym
from typing import Dict, List, Tuple


class DQNTorchMACRLModel(DQNTorchModel):
    pass


def build_q_model_and_distribution(
    policy: Policy,
    obs_space: gym.spaces.Space,
    action_space: gym.spaces.Space,
    config: AlgorithmConfigDict,
) -> Tuple[ModelV2, TorchDistributionWrapper]:
    """Build q_model and target_model for DQN

    Args:
        policy: The policy, which will use the model for optimization.
        obs_space (gym.spaces.Space): The policy's observation space.
        action_space (gym.spaces.Space): The policy's action space.
        config (AlgorithmConfigDict):

    Returns:
        (q_model, TorchCategorical)
            Note: The target q model will not be returned, just assigned to
            `policy.target_model`.
    """
    if not isinstance(action_space, gym.spaces.Discrete):
        raise UnsupportedSpaceException(
            "Action space {} is not supported for DQN.".format(action_space)
        )

    if config["hiddens"]:
        # try to infer the last layer size, otherwise fall back to 256
        num_outputs = ([256] + list(config["model"]["fcnet_hiddens"]))[-1]
        config["model"]["no_final_linear"] = True
    else:
        num_outputs = action_space.n

    # TODO(sven): Move option to add LayerNorm after each Dense
    #  generically into ModelCatalog.
    add_layer_norm = (
        isinstance(getattr(policy, "exploration", None), ParameterNoise)
        or config["exploration_config"]["type"] == "ParameterNoise"
    )

    model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel,
        name=Q_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
    )

    policy.target_model = ModelCatalog.get_model_v2(
        obs_space=obs_space,
        action_space=action_space,
        num_outputs=num_outputs,
        model_config=config["model"],
        framework="torch",
        model_interface=DQNTorchModel,
        name=Q_TARGET_SCOPE,
        q_hiddens=config["hiddens"],
        dueling=config["dueling"],
        num_atoms=config["num_atoms"],
        use_noisy=config["noisy"],
        v_min=config["v_min"],
        v_max=config["v_max"],
        sigma0=config["sigma0"],
        # TODO(sven): Move option to add LayerNorm after each Dense
        #  generically into ModelCatalog.
        add_layer_norm=add_layer_norm,
    )

    # Return a Torch TorchCategorical distribution where the temperature
    # parameter is partially binded to the configured value.
    temperature = config["categorical_distribution_temperature"]

    return model, get_torch_categorical_class_with_temperature(temperature)


DQNTorchMACRLPolicy = build_policy_class(
    name="DQNTorchPolicy",
    framework="torch",
    loss_fn=build_q_losses,
    get_default_config=lambda: ray.rllib.algorithms.dqn.dqn.DQNConfig(),
    make_model_and_action_dist=build_macrl_q_model_and_distribution,
    action_distribution_fn=get_distribution_inputs_and_class,
    stats_fn=build_q_stats,
    postprocess_fn=postprocess_nstep_and_prio,
    optimizer_fn=adam_optimizer,
    extra_grad_process_fn=grad_process_and_td_error_fn,
    extra_learn_fetches_fn=concat_multi_gpu_td_errors,
    extra_action_out_fn=extra_action_out_fn,
    before_init=setup_early_mixins,
    before_loss_init=before_loss_init,
    mixins=[
        TargetNetworkMixin,
        ComputeTDErrorMixin,
        LearningRateSchedule,
    ],
)