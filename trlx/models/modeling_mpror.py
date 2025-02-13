
from dataclasses import dataclass, field
from typing import Optional
from trlx.data.method_configs import register_method
from trlx.models.modeling_ppo import PPOConfig

# KL Controllers

# PPO Configs


@dataclass
@register_method
class MPRORConfig(PPOConfig):
    """
    Config for MPROR method

    The dataclass contains all the configs for PPO method + sampling config for MPROR RL. 

    """
    interval: Optional[int] = field(
        default=1,
        metadata={"help": "Do rollouts every interval steps."},
    )
    max_num_rollouts: Optional[int] = field(
        default=1024,
        metadata={"help": "Maximum number of rollouts to do per instance."},
    )
    max_rollout_length: Optional[int] = field(
        default=128,
        metadata={"help": "Maximum length of rollout trajectories."},
    )
    max_rollin_length: Optional[int] = field(
        default=100,
        metadata={"help": "Maximum length of rollout trajectories."},
    )
    exclude_first: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at first step (t=0)."},
    )
    exclude_last: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    must_include_first: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at first step (t=0)."},
    )
    must_include_last: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    use_sampling_curriculum: Optional[bool] = field(
        default=False,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    sampling_curr_coeff: Optional[int] = field(
        default=3,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    shifting_frac: Optional[float] = field(
        default=0.8,
        metadata={"help": "Do not rollout at last step (t=T)."},
    )
    propagate_gradients: Optional[bool] = field(
        default=False,
        metadata={"help": "Should we propagate gradient back to rollins?."},
    )
    discount_rollins: Optional[bool] = field(
        default=False,
        metadata={"help": "Should we Discount Rollins?."},
    )
