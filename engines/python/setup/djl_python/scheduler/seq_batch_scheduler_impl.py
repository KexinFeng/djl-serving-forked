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
from typing import Tuple, List

import torch

from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.batch import Batch
from djl_python.scheduler.step_generation import greedy_step_generate
from djl_python.scheduler.utils import compute_offsets, compute_position_ids, compute_attention_mask, \
    assemble_prefix_kv_cache


class GreedySeqBatchScheduler(SeqBatchScheduler):

    def init_forward(self,
                     input_ids,
                     request_ids,
                     kv_cache: Tuple = None,
                     save_kv_cache_path=None) -> Tuple[SeqBatcher, List[torch.Tensor]]:
        if input_ids.shape[0] != request_ids.shape[0] or len(
                request_ids.shape) != 2:
            raise Exception(
                "request_ids.shape does not match input_ids.shape or is illegal"
            )

        batch_size, init_seq_len = input_ids.shape
        init_offsets = compute_offsets(input_ids, self.config)
        attention_mask = compute_attention_mask(init_offsets, init_seq_len)
        position_ids = compute_position_ids(batch_size, init_seq_len, init_offsets, past_seq_len=0,
                                            repeat_offset=1)

        dummy_input_ids, position_ids, attention_mask, kv_cache = assemble_prefix_kv_cache(input_ids, position_ids,
                                                                                           attention_mask, kv_cache)

        # logits: [batch, sequence, vocab_dim]
        model_input = [input_ids, position_ids, attention_mask]
        logits, past_key_values, _ = self.lm_block.forward(model_input, past_key_values=kv_cache)

        # Create SeqBatcher
        if save_kv_cache_path:
            torch.save(past_key_values, save_kv_cache_path)

        batch = Batch(logits=logits[:, -1, :],
                      past_key_values=past_key_values)

        if kv_cache is not None:
            batch.nudge_to_squeeze_bubble_padding(init_offsets, kv_cache[0][0].shape[2])

        input_ids_list = []
        for i, input_id in enumerate(input_ids):
            to_append = input_id[init_offsets[i].item():]
            if kv_cache is not None:
                to_append = torch.concat([dummy_input_ids[i], to_append])
            input_ids_list.append(to_append)

        return SeqBatcher(batch, request_ids, init_offsets), input_ids_list

    def inference_call(self) -> Tuple[torch.Tensor, set]:
        batch = self.seq_batcher.batch

        # [batch, seq=1]
        output_ids = greedy_step_generate(batch.logits)
        assert len(output_ids.shape) == 2

        # prepare the next model_input
        position_ids = compute_position_ids(output_ids.shape[0], output_ids.shape[-1], self.seq_batcher.offsets,
                                            past_seq_len=self.seq_batcher.seq_len,
                                            repeat_offset=1)

        past_attention_mask = compute_attention_mask(self.seq_batcher.offsets, self.seq_batcher.seq_len + 1)

        # output: list(logits, past_kv, hidden_states), where logits: [batch, sequence, vocab_dim]
        logits, past_key_values, _ = self.lm_block.forward([output_ids, position_ids, past_attention_mask],
                                                           past_key_values=batch.past_key_values)

        # Create SeqBatcher
        last_logits = logits[:, -1, :]
        self.seq_batcher.batch = Batch(logits=last_logits,
                                       past_key_values=past_key_values)
        self.seq_batcher.seq_len += 1

        # exit check
        self.seq_batcher.exit_criteria(output_ids, self.config.max_seq_length,
                                       self.config.pad_token_id)

        return output_ids