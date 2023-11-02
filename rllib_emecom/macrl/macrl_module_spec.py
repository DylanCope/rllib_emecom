from rllib_emecom.macrl.comms.comms_spec import CommunicationSpec

from typing import Dict, Type, Any
from dataclasses import dataclass

import gymnasium as gym

from ray.rllib.core.rl_module.rl_module import (
    SingleAgentRLModuleSpec, RLModule
)
from ray.rllib.core.models.catalog import Catalog
from ray.rllib.utils.annotations import override
from ray.rllib.utils.serialization import (
    deserialize_type,
    serialize_type,
    gym_space_from_dict,
    gym_space_to_dict
)


@dataclass
class MACRLModuleConfig:
    """A utility config class to make it constructing MACRLModules easier.

    Args:
        comm_spec: The communication specification.
        observation_space: The observation space of the RLModule. This may differ
            from the observation space of the environment. For example, a discrete
            observation space of an environment, would usually correspond to a
            one-hot encoded observation space of the RLModule because of preprocessing.
        action_space: The action space of the RLModule.
        model_config_dict: The model config dict to use.
        catalog_class: The Catalog class to use.
    """
    comm_spec: CommunicationSpec = None
    observation_space: gym.Space = None
    action_space: gym.Space = None
    model_config_dict: Dict[str, Any] = None
    catalog_class: Type[Catalog] = None

    def get_catalog(self) -> Catalog:
        """Returns the catalog for this config."""
        return self.catalog_class(
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config_dict,
        )

    def to_dict(self):
        """Returns a serialized representation of the config.

        NOTE: This should be JSON-able. Users can test this by calling
            json.dumps(config.to_dict()).

        """
        catalog_class_path = (
            serialize_type(self.catalog_class) if self.catalog_class else ""
        )
        return {
            "observation_space": gym_space_to_dict(self.observation_space),
            "action_space": gym_space_to_dict(self.action_space),
            "model_config_dict": self.model_config_dict,
            "catalog_class_path": catalog_class_path,
            "comm_spec": self.comm_spec.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        """Creates a config from a serialized representation."""
        catalog_class = (
            None
            if d["catalog_class_path"] == ""
            else deserialize_type(d["catalog_class_path"])
        )
        return cls(
            comms_spec=CommunicationSpec.from_dict(d["comm_spec"]),
            observation_space=gym_space_from_dict(d["observation_space"]),
            action_space=gym_space_from_dict(d["action_space"]),
            model_config_dict=d["model_config_dict"],
            catalog_class=catalog_class,
        )


class MACRLModuleSpec(SingleAgentRLModuleSpec):

    def __init__(self, comm_spec: CommunicationSpec, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.comm_spec = comm_spec

    @override(SingleAgentRLModuleSpec)
    def get_rl_module_config(self) -> MACRLModuleConfig:
        """Returns the RLModule config for this spec."""
        return MACRLModuleConfig(
            comm_spec=self.comm_spec,
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config_dict=self.model_config_dict,
            catalog_class=self.catalog_class,
        )

    @classmethod
    def from_module(cls, module: RLModule) -> "MACRLModuleSpec":
        from ray.rllib.core.rl_module.marl_module import MultiAgentRLModule

        if isinstance(module, MultiAgentRLModule):
            raise ValueError(
                "MultiAgentRLModule cannot be converted to SingleAgentRLModuleSpec."
            )

        assert hasattr(module.config, 'comm_spec'), \
            "The module config must be a MACRLModuleConfig."

        return MACRLModuleSpec(
            comm_spec=module.config.comm_spec,
            module_class=type(module),
            observation_space=module.config.observation_space,
            action_space=module.config.action_space,
            model_config_dict=module.config.model_config_dict,
            catalog_class=module.config.catalog_class,
        )

    @classmethod
    def from_dict(cls, d):
        """Returns a single agent RLModule spec from a serialized representation."""
        module_class = deserialize_type(d["module_class"])

        module_config = MACRLModuleConfig.from_dict(d["module_config"])
        observation_space = module_config.observation_space
        action_space = module_config.action_space
        model_config_dict = module_config.model_config_dict
        catalog_class = module_config.catalog_class

        spec = MACRLModuleSpec(
            comm_spec=module_config.comm_spec,
            module_class=module_class,
            observation_space=observation_space,
            action_space=action_space,
            model_config_dict=model_config_dict,
            catalog_class=catalog_class,
        )
        return spec
