

import trlx.utils.logging as logging
from trlx.data.configs import TRLConfig
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer

logger = logging.get_logger(__name__)


@register_trainer
class AcceleratePPOMultiTrainer(AcceleratePPOTrainer):
    """PPO Accelerate Trainer"""
    def __init__(self, config: TRLConfig, **kwargs):
        """PPO Accelerate Trainer initialization

        Args:
            config: Config
        """
        config.method.num_rollouts = config.method.num_rollouts * config.method.max_num_rollouts
        if config.train.minibatch_size is None:
            config.train.minibatch_size = config.train.batch_size
        config.train.batch_size = config.train.batch_size * config.method.max_num_rollouts
        super().__init__(config, **kwargs)