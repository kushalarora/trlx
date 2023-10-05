

from typing import Any, Dict, List, Union
import numpy as np
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from trlx.models.modeling_mpror import MPRORConfig

from trlx.pipeline import BasePipeline, register_datapipeline
from torch.utils.data import IterableDataset
from trlx.data.default_configs import default_ppo_config

from trlx.data.method_configs import MethodConfig

@register_datapipeline
class MPRORPipeline(BasePipeline):
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
        config (`MPRORConfig`): config with method specific hyperparameters.
        is_eval (`bool`): whether the pipeline is used for evaluation or training.
    """

    def __init__(
        self,
        samples: Union[List[Dict[str, Any]], List[str]],
        max_prompt_length: int,
        tokenizer: PreTrainedTokenizer,
        add_special_tokens: bool = False,
        config: MethodConfig = None, 
        is_eval: bool = False,
        is_seq2seq: bool = False,
    ):
        super().__init__()

        if not config:
            config = default_ppo_config().method

        assert isinstance(config, MPRORConfig), "config must be an instance of MPRORConfig"

        assert isinstance(samples[0], dict), "prompts must be a list of dictionaries"

        metadata = samples
        prompts = [x.pop("prompt") for x in metadata]
        labels = [x.pop("label") for x in metadata]

        model_prompts = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length - config.max_rollin_length,
            add_special_tokens=add_special_tokens,
        )
        model_labels = tokenizer(
            prompts, truncation=True, padding=False, max_length=config.max_rollin_length,
            add_special_tokens=add_special_tokens,
        )

        new_prompts = []
        new_attentions = []
        if is_eval:
            new_prompts = prompts
        else:
            for (prompt, label, prompt_attention, label_attention) in zip(model_prompts['input_ids'], model_labels['input_ids'], 
                                        model_prompts['attention_mask'], model_labels['attention_mask']):
                num_label_tokens = len(label)
                num_rollouts = min(config.max_num_rollouts, 
                                    num_label_tokens // config.interval)
                max_rollin_length = min(num_label_tokens, config.max_rollin_length)

                if num_rollouts > 0 and not config.exclude_first:
                    new_prompts.append(prompt)
                    new_attentions.append(prompt_attention)
                    num_rollouts -= 1
                if num_rollouts > 0 and not config.exclude_last:
                    new_prompts.append(prompt + label)
                    new_attentions.append(prompt_attention + label_attention)
                    num_rollouts -= 1
                
                rollin_intervals = range(0, max_rollin_length, config.interval)
                for l in sorted(np.random.choice(rollin_intervals, num_rollouts, replace=False)):
                    rollin_prompt_suffix = label[:l]
                    rollin_attention_suffix = label_attention[:l]
                    rollout_prompt = prompt + rollin_prompt_suffix
                    rollout_attention = prompt_attention + rollin_attention_suffix
                    new_prompts.append(rollout_prompt)
                    new_attentions.append(rollout_attention)

                # shuffle_ix = np.random.permutation(len(new_prompts))
                # new_prompts = new_prompts[shuffle_ix]
        

        prompts_tokens = new_prompts
        attention_mask = new_attentions


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
            out = self.tokenizer.pad([{"input_ids": x["input_ids"]} for x in xs], return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs]

            return out

        # Since all data is already pre-processed, no need to have
        # multi-process data loading
        return DataLoader(
            self,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=0,
            drop_last=drop_last,
        )
