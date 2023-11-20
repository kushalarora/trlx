from typing import Any, Dict, List, Union
from torch.utils.data import DataLoader

from transformers import (
    PreTrainedTokenizer,
)

from trlx.pipeline import BasePipeline, register_datapipeline
from trlx.data.method_configs import MethodConfig
from trlx.data.default_configs import default_ppo_multi_config


@register_datapipeline
class PPOMultiPipeline(BasePipeline):
    """
    Dataloader which is used to supply prompts for either training or evaluation

    Args:
        prompts (`List[str]` or `List[Dict[str, Any]]`): list of raw text prompts or a dictionary with a required
            key `"prompt"` and extra information, that would be passed along the generation for that prompt as a
            keyword argument to a reward function.
        max_prompt_length (`int`): max length of the prompt, if exceeded the prompt will be truncated according to
            tokenizer's truncation setting.
        tokenizer (`transformers.PreTrainedTokenizer`): a tokenizer to tokenize prompts with.
        add_special_tokens (`bool`): whether to encode prompts with tokenizer's special tokens (passed directly
            into `tokenizer.encode`)
        config (`MethodConfig`): config with method specific hyperparameters.
        is_eval (`bool`): whether the pipeline is used for evaluation or training.
    """

    def __init__(
        self,
        prompts: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        total_steps: int,
        add_special_tokens: bool = False,
        config: MethodConfig = None,
        is_eval:bool = False
    ):
        super().__init__()

        if not config:
            config = default_ppo_multi_config().method
        self.is_eval = is_eval
        self.max_num_rollouts = 1 if is_eval else config.max_num_rollouts

        if isinstance(prompts[0], dict):
            metadata = prompts
            prompts = [x.pop("prompt") for x in metadata]
        else:
            metadata = [{}] * len(prompts)

        model_inputs = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length, add_special_tokens=add_special_tokens
        )

        prompts_tokens = model_inputs["input_ids"]
        attention_mask = model_inputs["attention_mask"]

        # new_prompts = []
        # new_attentions = []
        # new_metadata = []
        # if not is_eval:
        #     for (prompt, prompt_attention, md) in zip(prompts_tokens, attention_mask, metadata):
        #         num_rollouts = config.max_num_rollouts
        #         for _ in range(num_rollouts):
        #             new_prompts.append(prompt)
        #             new_attentions.append(prompt_attention)
        #             new_metadata.append(md)
        #     prompts_tokens = new_prompts
        #     attention_mask = new_attentions
        #     metadata = new_metadata

        self.tokenizer = tokenizer
        self.prompts = [
            {"input_ids": tokens, "attention_mask": mask, **metadata}
            for tokens, mask, metadata in zip(prompts_tokens, attention_mask, metadata)
        ]

    def __getitem__(self, ix: int):
        return self.prompts[ix]

    def __len__(self) -> int:
        return len(self.prompts)

    def create_loader(self, batch_size: int, shuffle=False, sampler=None, drop_last=False) -> DataLoader:
        def collate_fn(xs):
            input_ids = []
            if not self.is_eval:
                for x in xs:
                    for _ in range(self.max_num_rollouts):
                        input_ids.append({'input_ids': x["input_ids"]})
            else:
                for x in xs:
                    input_ids.append({'input_ids': x["input_ids"]})
            
            out = self.tokenizer.pad(input_ids, return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size//self.max_num_rollouts,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )