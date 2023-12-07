
import os
from time import time


from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


import trlx.utils.logging as logging
from trlx.data.accelerate_base_datatypes import PromptBatch
from trlx.data.configs import TRLConfig
from trlx.data.ppo_types import PPORLElement
from trlx.trainer import register_trainer
from trlx.trainer.accelerate_ppo_trainer import AcceleratePPOTrainer
from trlx.utils import Clock
from trlx.utils.modeling import gather_dict, logprobs_of_labels

logger = logging.get_logger(__name__)


@register_trainer
class AccelerateMPRORTrainer(AcceleratePPOTrainer):
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
        self.propagate_gradients = config.method.propagate_gradients
        self.discount_rollins = config.method.discount_rollins
        super().__init__(config, **kwargs)

    def decode(
        self,
        prompts: List[torch.LongTensor],
        samples: List[torch.LongTensor],
        prompt_sizes: torch.LongTensor = None,
        append_eos_token: bool = False,
    ) -> Tuple[List[str], List[str], List[str]]:
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """
        if prompt_sizes is None:
            # Assuming prompts were left-padded
            prompt_sizes = [prompts.shape[1]] * len(prompts)

        str_samples, str_prompts, str_outputs = [], [], []
        for prompt, sample, prompt_size in zip(prompts, samples, prompt_sizes):
            str_prompt = self.tokenizer.decode(prompt, skip_special_tokens=True)
            sample = self.tokenizer.decode(sample, skip_special_tokens=True)

            str_output = sample[len(str_prompt):]

            # Trim outputs up to `self.stop_sequences` if any are present
            trimmed = False
            if self.stop_sequences:
                for stop in self.stop_sequences:
                    stop_ix = str_output.find(stop)
                    if stop_ix >= 0:
                        str_output = str_output[:stop_ix].rstrip()
                        trimmed = True

            # Recover the last <eos> if it was present in the original sample
            # or add one if it was trimmed with `self.stop_sequences`.
            # When a generation ended due to `max_new_tokens` exhaustion,
            # only then <pad> or <eos> token would not be present in the original sample at the end
            if append_eos_token and (
                trimmed or sample[-1] == self.tokenizer.eos_token_id or sample[-1] == self.tokenizer.pad_token_id
            ):
                str_output += self.tokenizer.eos_token

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            sample = str_prompt + str_output
            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs


    def make_experience(self, num_rollouts: int = 1024, iter_count: int = 0):  # noqa:
        """Make experiences

        Takes `chunk_size` number of prompts from `prompt_iterator`, samples
        from the model and then computes the KL against a reference model. Finally it
        then appends PPOElements to trainer's `store`.

        Args:
            num_rollouts: Number of rollouts to generate
            iter_count: Total number of updates run (i.e. number of updates run for all batches & epochs)
        """
        logger.info("Collecting rollouts")
        tbar = logging.tqdm(
            total=num_rollouts,
            disable=os.environ.get("RANK", 0) != "0",
            desc=f"[rollout 0 / {num_rollouts}]",
            # Lower progress bar by 1 if we're in WARNING mode or above to avoid hiding high priority progress
            # bars (e.g. loss progress in trainers)
            position=logging.get_verbosity() >= logging.WARNING,
            # Leave progress bar if we're in INFO mode or lower to avoid spamming in suppressed verbosity levels
            leave=logging.get_verbosity() < logging.WARNING,
        )

        clock = Clock()
        ppo_rl_elements = []
        accumulated_stats = []
        score_diffs = []
        pct_rollouts = []

        while len(ppo_rl_elements) < num_rollouts:
            stats = {}
            # Get next batch in prompt dataset
            batch: PromptBatch = next(self.prompt_iterator)

            rollout_generate_time = time()

            # Generate samples from the language model (similar to using HuggingFace `generate` method)
            samples = self.generate(batch["augmented_input_ids"], batch["augmented_attention_mask"])

            stats["time/rollout_generate"] = time() - rollout_generate_time

            prompt_tensors = batch.input_ids if self.propagate_gradients else batch.augmented_input_ids

            augmented_prompts = batch.augmented_input_ids

            indexes = torch.tensor(batch.augmented_indexes, device=samples.device)
            to_mask_idxs = torch.arange(samples.shape[1], device=samples.device).unsqueeze(0) > samples.shape[1] - indexes.unsqueeze(1) - 1
            samples[to_mask_idxs] = self.tokenizer.pad_token_id

            device = samples.device
            # stats["mpror/pct_rollouts"] = batch.pct_rollouts
            stats["mpror/mean_pct_rollouts"] = np.mean(batch.pct_rollouts)
            stats["mpror/mean_indexes"] = np.median(batch.augmented_indexes)
            pct_rollouts.append(batch.pct_rollouts)

            prompt_sizes = torch.tensor([prompt_tensors.shape[1]] * len(prompt_tensors), device=device)
            padded_samples = self.accelerator.pad_across_processes(
                samples, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            padded_prompts = self.accelerator.pad_across_processes(
                prompt_tensors, dim=1, pad_index=self.tokenizer.eos_token_id, pad_first=False
            )
            gathered_samples = self.accelerator.gather(padded_samples)
            gathered_prompts = self.accelerator.gather(padded_prompts)
            gathered_prompt_sizes = self.accelerator.gather(prompt_sizes)
            gathered_indexes = self.accelerator.gather(indexes)
            metadata = gather_dict({k: v for k, v in batch.items() if k != "input_ids" and k != "attention_mask" and k != 'augmented_input_ids' and k != 'augmented_attention_mask'})

            if self.accelerator.is_main_process:
                all_str_samples, all_str_prompts, all_str_outputs = self.decode(
                    gathered_prompts, gathered_samples, gathered_prompt_sizes, append_eos_token=True
                )

                rollout_len = samples.shape[1]
                rollout_score_time = time()
                # reward_fn should return list of rewards at each token per sample
                # NOTE: all_scores[0][i] is the reward due to token (action) i in prompt + response (b/c of how kl is computed)
                all_scores = self.reward_fn(
                    samples=all_str_samples,
                    prompts=all_str_prompts,
                    outputs=all_str_outputs,
                    tokenizer=self.tokenizer,
                    **metadata,
                )

                if self.discount_rollins:
                    all_scores = [
                        torch.tensor(score * np.exp(-1*index/rollout_len), dtype=torch.float, device=device).view(
                            -1,
                        ) 
                        for score, index in zip(all_scores, gathered_indexes.tolist())
                    ]
                else:
                    all_scores = [
                        torch.tensor(score, dtype=torch.float, device=device).view(
                            -1,
                        ) 
                        for score in all_scores
                    ]

                # Pad 0 reward on the ends
                all_scores = pad_sequence(all_scores, batch_first=True, padding_value=-np.inf)
                max_len = torch.tensor(all_scores.shape[1], dtype=torch.long, device=device)

                stats["time/rollout_score"] = time() - rollout_score_time

                all_scores = list(all_scores.reshape(self.accelerator.num_processes, -1, max_len).unbind())
            else:
                all_scores = None
                max_len = torch.tensor(0, dtype=torch.long, device=device)

            if torch.distributed.is_initialized():
                torch.distributed.broadcast(max_len, 0)
                scores = torch.empty((len(samples), max_len), device=device)
                torch.distributed.scatter(scores, all_scores)
            else:
                scores = all_scores[0].clone().detach()
            scores_mask = scores != -np.inf

            str_samples, str_prompts, str_outputs = self.decode(prompt_tensors, samples, append_eos_token=True)
            batch_score_diffs = []
            prev_prompt_idx = -1
            first_idx_score = 0
            for prompt_idx,score in zip(batch.prompt_idxs, scores):
                if prompt_idx != prev_prompt_idx:
                    prev_prompt_idx = prompt_idx
                    first_idx_score = score
                    continue
                batch_score_diffs.append(float(score - first_idx_score))

            # stats['mpror/score_diffs'] = score_diffs
            stats['mpror/mean_score_diffs'] = np.mean(batch_score_diffs)
            score_diffs.append(batch_score_diffs)

            # Pad the sample outputs
            outputs = self.tokenizer(str_outputs).input_ids
            if self.config.model.model_arch_type == "seq2seq":
                # add <pad> to the start of the output
                for i in range(len(outputs)):
                    outputs[i] = [self.tokenizer.pad_token_id] + outputs[i]

            outputs = list(map(torch.LongTensor, outputs))
            maxsize = max(map(len, outputs))
            outputs = [
                F.pad(
                    output,
                    (0, maxsize - len(output)),
                    value=self.tokenizer.pad_token_id,
                )
                for output in outputs
            ]
            sample_outputs = torch.vstack(outputs).to(device)

            if self.config.method.cliprange_reward:
                scores = torch.clip(scores, -self.config.method.cliprange_reward, self.config.method.cliprange_reward)

            # store statistics of the initial rollout as reference
            if self.ref_mean is None:
                self.ref_mean, self.ref_std = (scores * scores_mask).sum(dim=1).mean(), (scores * scores_mask).sum(
                    dim=1
                ).std()
            all_scores_mean, all_scores_std = self.running_moments.update(torch.sum(scores * scores_mask, dim=1))
            stats["rollout_scores/mean"] = all_scores_mean.item()
            stats["rollout_scores/std"] = all_scores_std.item()
            stats["rollout_scores/running_mean"] = self.running_moments.mean.item()
            stats["rollout_scores/running_std"] = self.running_moments.std.item()

            if self.config.method.scale_reward == "running":
                scores /= self.running_moments.std
            elif self.config.method.scale_reward == "ref":
                scores /= self.ref_std

            # Precompute logprobs, values
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = batch.attention_mask.to(device)
                prompt_tensors = batch.input_ids.to(device)
                decoder_attention_mask = sample_outputs.not_equal(self.tokenizer.pad_token_id)
                decoder_attention_mask[:, 0] = 1
                with torch.no_grad():
                    outputs = self.model(
                        input_ids=prompt_tensors,
                        attention_mask=attention_mask,
                        decoder_input_ids=sample_outputs,
                        decoder_attention_mask=decoder_attention_mask,
                    )
                    logits = outputs.logits
                    values = outputs.value
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            input_ids=prompt_tensors,
                            attention_mask=attention_mask,
                            decoder_input_ids=sample_outputs,
                            decoder_attention_mask=decoder_attention_mask,
                            return_dict=True,
                        ).logits
            else:
                all_tokens = torch.cat((prompt_tensors.to(device), sample_outputs), dim=1)
                attention_mask = all_tokens.not_equal(self.tokenizer.pad_token_id).long().to(device)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                with torch.no_grad():
                    logits, *_, values = self.model(
                        all_tokens, attention_mask=attention_mask, position_ids=position_ids
                    )
                    # TODO(dahoas): When hydra model works need to also support generation on hydra head
                    if hasattr(self.model, "frozen_head") or self.model.peft_type:
                        ref_logits = self.model.forward_hydra(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                    else:
                        ref_logits = self.ref_model(
                            all_tokens,
                            attention_mask=attention_mask,
                            position_ids=position_ids,
                            return_dict=True,
                        ).logits
                        ref_logits = ref_logits.to(device)

            if self.config.model.model_arch_type == "seq2seq":
                logprobs = logprobs_of_labels(logits[:, :-1, :], sample_outputs[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], sample_outputs[:, 1:])
            else:
                # NOTE: logprob[i] is (log)prob at which all_token[i+1] was sampled
                logprobs = logprobs_of_labels(logits[:, :-1, :], all_tokens[:, 1:])
                ref_logprobs = logprobs_of_labels(ref_logits[:, :-1, :], all_tokens[:, 1:])

            n_samples: int = samples.shape[0]

            # Estimate the KL divergence between the model and reference model
            if self.config.model.model_arch_type == "seq2seq":
                attention_mask = sample_outputs != self.tokenizer.pad_token_id
                start = 0
            else:
                start = prompt_tensors.shape[1] - 1

            log_ratio = (logprobs - ref_logprobs) * attention_mask[:, :-1]
            kl = log_ratio.exp() - 1 - log_ratio
            mean_kl_per_token = kl.mean()
            mean_kl = kl.sum(1).mean()

            logprobs = logprobs.cpu()
            ref_logprobs = ref_logprobs.cpu()
            prompt_tensors = prompt_tensors.cpu()
            sample_outputs = sample_outputs.cpu()
            values = values.cpu()[:, :-1]

            # Get the logprobs and values, for tokens that are not padding,
            # from the end of the prompt up to the <eos> token, while also including the latter
            # (these are taken from the student model and not the reference model)
            ends = start + attention_mask[:, start:].sum(1) + 1
            all_values = [values[ix, start : ends[ix]] for ix in range(n_samples)]
            all_logprobs = [logprobs[ix, start : ends[ix]] for ix in range(n_samples)]

            kl_penalty = self.kl_ctl.value * -log_ratio.cpu()
            kl_penalty = [xs[start : ends[ix]] for ix, xs in enumerate(kl_penalty)]

            rollout_count = 0

            for sample_idx in range(n_samples):
                rewards = kl_penalty[sample_idx]
                # Then add in rewards
                if scores.shape[1] == 1:
                    # NOTE: Final reward given at EOS token following HHH practice
                    rewards[-1] += scores[sample_idx][0].cpu()
                else:
                    score = scores[sample_idx]
                    score_right_padding = torch.sum(scores_mask[sample_idx])
                    score = score[:score_right_padding].cpu()
                    p_score = torch.zeros_like(rewards)
                    p_score[: score.shape[0]] += score
                    rewards += p_score

                ppo_rl_elements.append(
                    PPORLElement(
                        query_tensor=prompt_tensors[sample_idx],
                        response_tensor=sample_outputs[sample_idx],
                        logprobs=all_logprobs[sample_idx],
                        values=all_values[sample_idx],
                        rewards=rewards,
                    )
                )

                rollout_count += 1

            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(mean_kl, torch.distributed.ReduceOp.AVG)

            stats["time/rollout_time"] = clock.tick()
            stats["policy/sqrt_kl"] = torch.sqrt(mean_kl).item()
            stats["policy/kl_per_token"] = torch.sqrt(mean_kl_per_token).item()
            accumulated_stats.append(stats)

            tbar.set_description(f"[rollout {len(ppo_rl_elements)} / {num_rollouts}]")
            tbar.update(min(rollout_count, num_rollouts))
        tbar.close()

        stats = {k: sum([xs[k] for xs in accumulated_stats]) / len(accumulated_stats) for k in stats}
        stats["kl_ctl_value"] = self.kl_ctl.value
        stats["mpror/pct_rollouts"] = np.array(pct_rollouts)
        stats["mpror/score_diffs"] = np.array(score_diffs)
        self.mean_kl = stats["policy/sqrt_kl"] ** 2
        self.accelerator.log(stats, step=iter_count)

        # Push samples and rewards to trainer's rollout storage
        self.push_to_store(ppo_rl_elements)