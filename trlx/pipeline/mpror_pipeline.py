

from typing import Any, Dict, List, Union
import numpy as np
from transformers import PreTrainedTokenizer
from torch.utils.data import DataLoader
from trlx.models.modeling_mpror import MPRORConfig

from trlx.pipeline import BasePipeline, register_datapipeline
from torch.utils.data import IterableDataset
from trlx.data.default_configs import default_mpror_config

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
        total_epochs: int,
        add_special_tokens: bool = False,
        config: MethodConfig = None, 
        is_eval: bool = False,
        is_seq2seq: bool = False,
    ):
        super().__init__()

        if not config:
            config = default_mpror_config().method

        assert isinstance(config, MPRORConfig), "config must be an instance of MPRORConfig"

        assert isinstance(samples[0], dict), "prompts must be a list of dictionaries"

        self.config = config
        self.is_eval = is_eval
        self.is_seq2seq = is_seq2seq
        self.max_num_rollouts = 1 if is_eval else config.max_num_rollouts
        self.add_special_tokens = add_special_tokens
        self.total_epochs = total_epochs
        self.curr_epoch = -1

        metadata = samples
        prompts = [x["prompt"] for x in metadata]
        labels = [" "  + x["label"] for x in metadata]

        model_prompts = tokenizer(
            prompts, truncation=True, padding=False, max_length=max_prompt_length,
            add_special_tokens=add_special_tokens,
        )
       
        new_prompts = []
        new_attentions = []
        mpror_prompts_and_labels = []
        prompts_tokens = model_prompts['input_ids']
        attention_mask = model_prompts['attention_mask']

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
            labels = [" "  + x["label"] for x in xs]

            model_labels = self.tokenizer(labels, truncation=True, padding=False, 
                            max_length=self.config.max_rollin_length, add_special_tokens=self.add_special_tokens,)

            prompts_tokens = [x["input_ids"] for x in xs]
            indexes = [0] * len(prompts_tokens)

            label_tokens = model_labels['input_ids']
            if not self.is_eval:
                new_prompts = []
                indexes = []
                for (prompt, label) in zip(prompts_tokens, label_tokens):
                    num_label_tokens = len(label)
                    num_rollouts = min(self.config.max_num_rollouts, 
                                        num_label_tokens // self.config.interval)

                    intervals = []
                    start_index = 0
                    end_index = num_label_tokens
                    if self.config.exclude_first:
                        start_index += 1
                    elif self.config.must_include_first:
                        intervals.append(0)
                        start_index += 1
                        num_rollouts -= 1

                    if self.config.exclude_last:
                        end_index -= 1
                    elif self.config.must_include_last:
                        intervals.append(end_index)
                        end_index -= 1
                        num_rollouts -= 1

                    max_rollin_length = min(end_index, self.config.max_rollin_length)

                    rollin_intervals = range(start_index, max_rollin_length, self.config.interval)
                    interval_length = len(rollin_intervals)
                    dist_weights = np.array([np.sin(np.pi * i/(interval_length - 1)/2 + 
                                            np.pi * self.curr_epoch/(self.total_epochs - 1)/2)
                                            for i in range(interval_length)])
                    dist_weights = np.array(dist_weights) / np.sum(dist_weights)
                    intervals += list(np.random.choice(rollin_intervals, num_rollouts, replace=False, p=dist_weights))
                    for l in sorted(intervals):
                        rollin_prompt_suffix = label[:l]
                        rollout_prompt = prompt + rollin_prompt_suffix
                        new_prompts.append(rollout_prompt)
                        indexes.append(l)

                prompts_tokens = new_prompts

            out = self.tokenizer.pad([{"input_ids": x} for idx, x in zip(indexes, prompts_tokens)], return_tensors="pt")

            for key in xs[0]:
                if key != "input_ids" and key != "attention_mask":
                    out[key] = [x[key] for x in xs for _ in range(self.max_num_rollouts)]
            return out

        self.curr_epoch += 1

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