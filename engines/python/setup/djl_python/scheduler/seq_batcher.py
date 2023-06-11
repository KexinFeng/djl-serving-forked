#!/usr/bin/env python
#
# Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file
# except in compliance with the License. A copy of the License is located at
#
# http://aws.amazon.com/apache2.0/
#
# or in the "LICENSE.txt" file accompanying this file. This file is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, express or implied. See the License for
# the specific language governing permissions and limitations under the License.
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Union, Tuple, List, Any
from abc import ABC, abstractmethod

from djl_python.scheduler.batch import Batch, ContrastiveBatch, BeamBatch
from djl_python.scheduler.lm_block import LMBlock
import torch
from torch.nn.functional import normalize, softmax

from djl_python.scheduler.step_generation import greedy_step_generate, contrastive_step_generate
from djl_python.scheduler.utils import compute_offsets, compute_attention_mask, compute_position_ids, \
    assemble_prefix_kv_cache
from djl_python.scheduler import SearchConfig


class SeqBatcher(ABC):
    """
    This is specific to a search algorithm, which frees the scheduler from being searching
    algorithm specific. In the future, user may provide their own autoregressive searching algorithm by overwriting
    the abstract classes.
    """

    def __init__(self, batch: Batch, request_uids: torch.Tensor,
                 offsets: torch.Tensor,
                 search_configs: defaultdict[Any,
                                             SearchConfig], lm_block: LMBlock):
        self.batch = batch
        self.request_uids = request_uids
        self.offsets = offsets
        self.search_configs = search_configs
        self.exit_index = set()
        self.lm_block = lm_block

        self.batch_size, _, self.seq_len, _ = batch.past_key_values[0][0].size(
        )

    @classmethod
    @abstractmethod
    @torch.no_grad()
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[str]]]:
        pass

    @staticmethod
    @abstractmethod
    def get_batch_cls():
        pass

    @abstractmethod
    def inference_call(self):
        pass

    @torch.no_grad()
    def add_batch(self, seq_batcher: SeqBatcher):
        if self.lm_block != seq_batcher.lm_block:
            raise "lm_blocks are not the same instance, not mergable"

        self._merge_symmetric(self, seq_batcher)

    def _merge_symmetric(self, seq_batcher1: SeqBatcher,
                         seq_batcher2: SeqBatcher):
        seq_delta = seq_batcher1.seq_len - seq_batcher2.seq_len
        if seq_delta < 0:
            seq_batcher1, seq_batcher2 = seq_batcher2, seq_batcher1
            seq_delta = -seq_delta

        # merge batches
        self.batch = seq_batcher1.batch.merge(seq_batcher2.batch, seq_delta)

        # update other batch control variables
        self.batch_size = seq_batcher1.batch_size + seq_batcher2.batch_size
        self.request_uids = torch.cat(
            [seq_batcher1.request_uids, seq_batcher2.request_uids], dim=0)
        self.offsets = torch.cat(
            [seq_batcher1.offsets, seq_batcher2.offsets + seq_delta], dim=0)
        self.seq_len = max(seq_batcher1.seq_len, seq_batcher2.seq_len)
        seq_batcher1.search_configs.update(seq_batcher2.search_configs)
        self.search_configs = seq_batcher1.search_configs

    @torch.no_grad()
    def collect_and_trim(self) -> None:
        if len(self.exit_index) == 0:
            return

        # find the batch indices of the non-finished requests.
        batch_size = self.request_uids.shape[0]
        keep_indices = torch.tensor(
            list(set(range(batch_size)) - self.exit_index),
            dtype=torch.int64,
            device=self.offsets.device)

        # if all the requests finished generating sequences, then reset the batch and return
        if len(keep_indices) == 0:
            self.request_uids = torch.empty([0, 1],
                                            dtype=self.request_uids.dtype,
                                            device=self.request_uids.device)
            self.offsets = torch.empty([0, 1],
                                       dtype=self.offsets.dtype,
                                       device=self.offsets.device)
            self.search_configs.clear()

            self.batch = None
            self.batch_size = 0
            self.seq_len = 0
        else:
            for idx in self.exit_index:
                del self.search_configs[self.request_uids[idx].item()]
            self.request_uids = self.request_uids[keep_indices]
            self.offsets = self.offsets[keep_indices]
            trim_seq_len = torch.min(self.offsets, dim=0).values.item()
            self.offsets.sub_(trim_seq_len)

            self.batch.trim(keep_indices, trim_seq_len)
            self.batch_size -= len(self.exit_index)
            self.seq_len -= trim_seq_len

        self.exit_index = set()

    def exit_criteria(self, output_ids: torch.Tensor, search_configs):
        for i, (output_id, request_uid, offset) in enumerate(
                zip(
                    output_ids.view(-1).tolist(),
                    self.request_uids.view(-1).tolist(),
                    self.offsets.view(-1).tolist())):
            if self.seq_len - offset >= search_configs[request_uid].max_seqlen \
                    or output_id == search_configs[request_uid].eos_token_id:
                if i not in self.exit_index:
                    self.exit_index.add(i)

    def seq_complete(self) -> bool:
        return len(self.exit_index) > 0

    def is_empty(self) -> bool:
        return self.batch is None


class GreedySeqBatcher(SeqBatcher):

    @classmethod
    @torch.no_grad()
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[str]]]:

        if input_ids.shape[0] != request_uids.shape[0] or len(
                request_uids.shape) != 2:
            raise Exception(
                "request_uids.shape does not match input_ids.shape or is illegal"
            )

        initial_offsets = compute_offsets(input_ids, [
            search_configs[r].pad_token_id
            for r in request_uids.view(-1).tolist()
        ])
        attention_mask = compute_attention_mask(initial_offsets,
                                                input_ids.shape[-1])
        position_ids = compute_position_ids(input_ids.shape[0],
                                            input_ids.shape[1],
                                            initial_offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        # Handle the kv_cache
        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(
            input_ids, position_ids, attention_mask, kv_cache)

        # Forward call
        model_input = [input_ids, position_ids, attention_mask]
        logits, past_key_values, past_hidden_states = lm_block.forward(
            model_input, past_key_values=kv_cache)
        last_logits = logits[:, -1, :]

        # Save kv_cache of input_ids
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        # Generate next token and batch
        next_input_ids = greedy_step_generate(
            last_logits).indices  # [batch, 1]
        batch = Batch(next_input_ids=next_input_ids,
                      past_key_values=past_key_values)
        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(initial_offsets,
                                                  kv_cache[0][0].shape[2])

        # Output
        output_ids_list = []
        for i, (input_id,
                offset) in enumerate(zip(input_ids.tolist(), initial_offsets)):
            to_append = input_id[offset:]
            if kv_cache is not None:
                to_append = dummy_input_ids[i].tolist() + to_append
            output_ids_list.append(to_append)

        return cls(batch, request_uids, initial_offsets, search_configs,
                   lm_block), output_ids_list

    @torch.no_grad()
    def inference_call(self) -> List[List[int]]:
        batch = self.batch

        # [batch, seq=1]
        output_ids = batch.next_input_ids
        assert len(output_ids.shape) == 2

        # Prepare the next model_input
        position_ids = compute_position_ids(output_ids.shape[0],
                                            output_ids.shape[-1],
                                            self.offsets,
                                            past_seq_len=self.seq_len,
                                            repeat_offset=1)

        past_attention_mask = compute_attention_mask(self.offsets,
                                                     self.seq_len + 1)

        # Forward pass
        logits, past_key_values, _ = self.lm_block.forward(
            [output_ids, position_ids, past_attention_mask],
            past_key_values=batch.past_key_values)

        # Create SeqBatcher
        last_logits = logits[:, -1, :]  # logits: [batch, sequence, vocab_dim]
        next_input_ids = greedy_step_generate(
            last_logits).indices  # [batch, 1]
        self.batch = Batch(past_key_values=past_key_values,
                           next_input_ids=next_input_ids)
        self.seq_len += 1

        # Exit check (It is ok that next_input_ids is not output here, since the exit criteria doesn't check it but
        # output_ids instead.)
        self.exit_criteria(output_ids, self.search_configs)

        return output_ids.tolist()

    @staticmethod
    def get_batch_cls():
        return Batch


class ContrastiveSeqBatcher(SeqBatcher):

    @classmethod
    @torch.no_grad()
    def init_forward(
            cls,
            input_ids: torch.tensor,
            request_uids: torch.tensor,
            lm_block: LMBlock,
            search_configs: defaultdict[Any, SearchConfig],
            kv_cache: Union[Tuple, None] = None,
            save_kv_cache_path=None) -> Tuple[SeqBatcher, List[List[str]]]:

        if input_ids.shape[0] != request_uids.shape[0] or len(
                request_uids.shape) != 2:
            raise Exception(
                "request_uids.shape does not match input_ids.shape or is illegal"
            )

        initial_offsets = compute_offsets(input_ids, [
            search_configs[r].pad_token_id
            for r in request_uids.view(-1).tolist()
        ])
        attention_mask = compute_attention_mask(initial_offsets,
                                                input_ids.shape[-1])
        position_ids = compute_position_ids(input_ids.shape[0],
                                            input_ids.shape[1],
                                            initial_offsets,
                                            past_seq_len=0,
                                            repeat_offset=1)
        # Handle the kv_cache
        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(
            input_ids, position_ids, attention_mask, kv_cache)

        # Forward call
        model_input = [input_ids, position_ids, attention_mask]
        logits, past_key_values, _ = lm_block.forward(
            model_input, past_key_values=kv_cache)
        last_logits = logits[:, -1, :]

        # Save kv_cache of input_ids
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        # Special handling for contrastive search below
        if kv_cache is not None:
            past_hidden_states = torch.concat([
                torch.zeros(input_ids.shape[0],
                            kv_cache[0][0].shape[2],
                            past_hidden_states.shape[-1],
                            dtype=past_hidden_states.dtype,
                            device=past_hidden_states.device),
                past_hidden_states
            ],
                                              dim=1)

        # Generate next token and batch
        topk = search_configs["non_exist_key"].topk
        # [batch, vocab_size=50257]
        last_probs = softmax(last_logits, dim=1)
        # [batch, topk]
        top_k_probs, top_k_ids = greedy_step_generate(
            last_probs, topk)
        batch = cls.get_batch_cls()(next_input_ids=top_k_ids,
                                    past_key_values=past_key_values,
                                    past_hidden_states=past_hidden_states,
                                    top_k_probs=top_k_probs)
        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(initial_offsets,
                                                  kv_cache[0][0].shape[2])
        # Output ids
        output_ids_list = []
        for i, (input_id,
                offset) in enumerate(zip(input_ids.tolist(), initial_offsets)):
            to_append = input_id[offset:]
            if kv_cache is not None:
                to_append = dummy_input_ids[i].tolist() + to_append
            output_ids_list.append(to_append)

        return cls(batch, request_uids, initial_offsets, search_configs,
                   lm_block), output_ids_list

    @torch.no_grad()
    def inference_call(self) -> List[List[int]]:
        batch = self.batch
        config = self.search_configs["non_exist_key"]

        # [batch, topK]
        top_k_ids = batch.next_input_ids

        '''
        Prepare candidate model input
        '''
        # [batch, topK] -> [batch * [topK]] -> [[batch * [topK]], seqLength=1]
        candidate_input_ids = top_k_ids.view(-1, 1)
        assert candidate_input_ids.dtype == torch.int64
        assert len(candidate_input_ids.shape) == 2

        # [batch, heads, seq_past, feature] -> [batch * topK, head, seq_past, feature]
        k_copy_past_key_values = []
        for k, v in batch.past_key_values:
            k_new = torch.repeat_interleave(k, dim=0, repeats=config.topk)
            v_new = torch.repeat_interleave(v, dim=0, repeats=config.topk)
            k_copy_past_key_values.append((k_new, v_new))
        k_copy_past_key_values = tuple(k_copy_past_key_values)

        # [batch, seq_past] -> [batch * topK, seq_past] -> [batch * topK, seq_past + 1]
        batch_size = top_k_ids.shape[0]
        k_copy_past_attention_mask = compute_attention_mask(
            offsets=self.offsets,
            seq_len=self.seq_len + 1,
            repeat_offset=config.topk)
        candidate_position_ids = compute_position_ids(
            candidate_input_ids.shape[0],
            candidate_input_ids.shape[1],
            self.offsets,
            past_seq_len=self.seq_len,
            repeat_offset=config.topk)

        # [batch * topK, ..., seq_past + 1, ...]
        candidate_logits, candidate_past_key_values, _ = self.lm_block.forward(
            [candidate_input_ids, candidate_position_ids, k_copy_past_attention_mask],
            k_copy_past_key_values)

        output_ids, select = contrastive_step_generate(
            top_k_ids=top_k_ids,
            top_k_probs=batch.top_k_probs,
            top_k_hidden_states=self.lm_block.embedding(candidate_input_ids),
            context_hidden_states=self.lm_block.embedding(batch.past_output_ids),
            offsets=self.offsets,
            alpha=config.alpha)

        '''
        Select from the topk candidates and generate output and the new batch
        '''
        logits_dim = candidate_logits.shape[-1]
        _, num_heads, _, kv_dim = batch.past_key_values[0][0].shape
        past_seq_len = self.seq_len
        hidden_dim = batch.past_hidden_states.shape[-1]

        # [batch, 1]
        a_range = torch.arange(batch_size)
        next_logits = candidate_logits.view(batch_size, config.topk,
                                            logits_dim)[a_range, select]

        next_past_key_values = []
        for k, v in candidate_past_key_values:
            k_new = k.view(batch_size, config.topk, num_heads,
                           past_seq_len + 1, kv_dim)[a_range, select]
            v_new = v.view(batch_size, config.topk, num_heads,
                           past_seq_len + 1, kv_dim)[a_range, select]
            next_past_key_values.append((k_new, v_new))
        next_past_key_values = tuple(next_past_key_values)
        # [batch, past_seq + 1]
        next_output_ids = torch.concat(
            [batch.past_output_ids, output_ids], dim=1)

        self.seq_len += 1
        # [batch, vocab_size]
        next_probs = softmax(next_logits, dim=1)
        # [batch, topk]
        top_k_probs, top_k_ids = greedy_step_generate(
            next_probs, config.topk)  
        self.batch = ContrastiveBatch(next_input_ids=top_k_ids,
                                      past_key_values=next_past_key_values,
                                      past_output_ids=next_output_ids,
                                      top_k_probs=top_k_probs)

        # Exit
        self.exit_criteria(output_ids, self.search_configs)

        return output_ids.tolist()

    @staticmethod
    def get_batch_cls():
        return ContrastiveBatch


class BeamSeqBatcher(SeqBatcher):

    @classmethod
    def get_batch_cls(cls):
        return BeamBatch

    def inference_call(self):
        print("Reach here! Process the logits")
        pass
