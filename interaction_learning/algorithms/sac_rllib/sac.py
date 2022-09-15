import logging
from typing import Type, Dict, Any, Optional, Union

from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.dqn.dqn import DQN
from ray.rllib.policy.policy import Policy
from ray.rllib.utils.annotations import override
from ray.rllib.utils.deprecation import (
    DEPRECATED_VALUE,
    deprecation_warning,
    Deprecated,
)
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import AlgorithmConfigDict

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()

logger = logging.getLogger(__name__)


class SACConfig(AlgorithmConfig):

    def __init__(self, algo_class=None):
        super().__init__(algo_class=algo_class or SAC)
        # fmt: off
        # __sphinx_doc_begin__
        # SAC-specific config settings.
        self.twin_q = True
        self.q_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define custom Q-model(s).
            "custom_model_config": {},
        }
        self.policy_model_config = {
            "fcnet_hiddens": [256, 256],
            "fcnet_activation": "relu",
            "post_fcnet_hiddens": [],
            "post_fcnet_activation": None,
            "custom_model": None,  # Use this to define a custom policy model.
            "custom_model_config": {},
        }
        self.clip_actions = False
        self.tau = 5e-3
        self.initial_alpha = 1.0
        self.target_entropy = "auto"
        self.n_step = 1
        self.replay_buffer_config = {
            "_enable_replay_buffer_api": True,
            "type": "MultiAgentPrioritizedReplayBuffer",
            "capacity": int(1e6),
            # If True prioritized replay buffer will be used.
            "prioritized_replay": False,
            "prioritized_replay_alpha": 0.6,
            "prioritized_replay_beta": 0.4,
            "prioritized_replay_eps": 1e-6,
            # Whether to compute priorities already on the remote worker side.
            "worker_side_prioritization": False,
        }
        self.store_buffer_in_checkpoints = False
        self.training_intensity = None
        self.optimization = {
            "actor_learning_rate": 3e-4,
            "critic_learning_rate": 3e-4,
            "entropy_learning_rate": 3e-4,
        }
        self.grad_clip = None
        self.target_network_update_freq = 0

        # .rollout()
        self.rollout_fragment_length = 1
        self.compress_observations = False

        # .training()
        self.train_batch_size = 256
        # Number of timesteps to collect from rollout workers before we start
        # sampling from replay buffers for learning. Whether we count this in agent
        # steps  or environment steps depends on config["multiagent"]["count_steps_by"].
        self.num_steps_sampled_before_learning_starts = 1500

        # .reporting()
        self.min_time_s_per_iteration = 1
        self.min_sample_timesteps_per_iteration = 100
        # __sphinx_doc_end__
        # fmt: on

        self._deterministic_loss = False
        self._use_beta_distribution = False

        self.use_state_preprocessor = DEPRECATED_VALUE
        self.worker_side_prioritization = DEPRECATED_VALUE

    @override(AlgorithmConfig)
    def training(
        self,
        *,
        twin_q: Optional[bool] = None,
        q_model_config: Optional[Dict[str, Any]] = None,
        policy_model_config: Optional[Dict[str, Any]] = None,
        tau: Optional[float] = None,
        initial_alpha: Optional[float] = None,
        target_entropy: Optional[Union[str, float]] = None,
        n_step: Optional[int] = None,
        store_buffer_in_checkpoints: Optional[bool] = None,
        replay_buffer_config: Optional[Dict[str, Any]] = None,
        training_intensity: Optional[float] = None,
        clip_actions: Optional[bool] = None,
        grad_clip: Optional[float] = None,
        optimization_config: Optional[Dict[str, Any]] = None,
        target_network_update_freq: Optional[int] = None,
        _deterministic_loss: Optional[bool] = None,
        _use_beta_distribution: Optional[bool] = None,
        num_steps_sampled_before_learning_starts: Optional[int] = None,
        **kwargs,
    ) -> "SACConfig":
        # Pass kwargs onto super's `training()` method.
        super().training(**kwargs)

        if twin_q is not None:
            self.twin_q = twin_q
        if q_model_config is not None:
            self.q_model_config = q_model_config
        if policy_model_config is not None:
            self.policy_model_config = policy_model_config
        if tau is not None:
            self.tau = tau
        if initial_alpha is not None:
            self.initial_alpha = initial_alpha
        if target_entropy is not None:
            self.target_entropy = target_entropy
        if n_step is not None:
            self.n_step = n_step
        if store_buffer_in_checkpoints is not None:
            self.store_buffer_in_checkpoints = store_buffer_in_checkpoints
        if replay_buffer_config is not None:
            self.replay_buffer_config = replay_buffer_config
        if training_intensity is not None:
            self.training_intensity = training_intensity
        if clip_actions is not None:
            self.clip_actions = clip_actions
        if grad_clip is not None:
            self.grad_clip = grad_clip
        if optimization_config is not None:
            self.optimization = optimization_config
        if target_network_update_freq is not None:
            self.target_network_update_freq = target_network_update_freq
        if _deterministic_loss is not None:
            self._deterministic_loss = _deterministic_loss
        if _use_beta_distribution is not None:
            self._use_beta_distribution = _use_beta_distribution
        if num_steps_sampled_before_learning_starts is not None:
            self.num_steps_sampled_before_learning_starts = (
                num_steps_sampled_before_learning_starts
            )

        return self


class SAC(DQN):
    """Soft Actor Critic (SAC) Algorithm class.
    This file defines the distributed Algorithm class for the soft actor critic
    algorithm.
    See `sac_[tf|torch]_policy.py` for the definition of the policy loss.
    Detailed documentation:
    https://docs.ray.io/en/master/rllib-algorithms.html#sac
    """

    def __init__(self, *args, **kwargs):
        self._allow_unknown_subkeys += ["policy_model_config", "q_model_config"]
        super().__init__(*args, **kwargs)

    @classmethod
    @override(DQN)
    def get_default_config(cls) -> AlgorithmConfigDict:
        return SACConfig().to_dict()

    @override(DQN)
    def validate_config(self, config: AlgorithmConfigDict) -> None:
        # Call super's validation method.
        super().validate_config(config)

        if config["use_state_preprocessor"] != DEPRECATED_VALUE:
            deprecation_warning(old="config['use_state_preprocessor']", error=False)
            config["use_state_preprocessor"] = DEPRECATED_VALUE

        if config.get("policy_model", DEPRECATED_VALUE) != DEPRECATED_VALUE:
            deprecation_warning(
                old="config['policy_model']",
                new="config['policy_model_config']",
                error=False,
            )
            config["policy_model_config"] = config["policy_model"]

        if config.get("Q_model", DEPRECATED_VALUE) != DEPRECATED_VALUE:
            deprecation_warning(
                old="config['Q_model']",
                new="config['q_model_config']",
                error=False,
            )
            config["q_model_config"] = config["Q_model"]

        if config["grad_clip"] is not None and config["grad_clip"] <= 0.0:
            raise ValueError("`grad_clip` value must be > 0.0!")

        if config["framework"] in ["tf", "tf2", "tfe"] and tfp is None:
            logger.warning(
                "You need `tensorflow_probability` in order to run SAC! "
                "Install it via `pip install tensorflow_probability`. Your "
                f"tf.__version__={tf.__version__ if tf else None}."
                "Trying to import tfp results in the following error:"
            )
            try_import_tfp(error=True)

    @override(DQN)
    def get_default_policy_class(self, config: AlgorithmConfigDict) -> Type[Policy]:
        if config["framework"] == "torch":
            from ray.rllib.algorithms.sac.sac_torch_policy import SACTorchPolicy

            return SACTorchPolicy
        else:
            return SACTFPolicy


# Deprecated: Use ray.rllib.algorithms.sac.SACConfig instead!
class _deprecated_default_config(dict):
    def __init__(self):
        super().__init__(SACConfig().to_dict())

    @Deprecated(
        old="ray.rllib.algorithms.sac.sac::DEFAULT_CONFIG",
        new="ray.rllib.algorithms.sac.sac::SACConfig(...)",
        error=False,
    )
    def __getitem__(self, item):
        return super().__getitem__(item)


DEFAULT_CONFIG = _deprecated_default_config()
