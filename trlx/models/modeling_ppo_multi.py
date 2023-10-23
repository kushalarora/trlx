
from dataclasses import dataclass, field
from typing import Optional
from trlx.data.method_configs import register_method
from trlx.models.modeling_ppo import PPOConfig

# KL Controllers

# PPO Configs


@dataclass
@register_method
class PPOMultiConfig(PPOConfig):
    """
    Config for PPOMultiConfig method

    The dataclass contains all the configs for PPO method + max_num_rollouts for multiple rollouts. 

    """
    max_num_rollouts: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum number of rollouts to do per instance."},
    )