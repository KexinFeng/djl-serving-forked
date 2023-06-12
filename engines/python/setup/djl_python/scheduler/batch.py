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

import torch
from djl_python.scheduler.utils import merge_tensors, trim_tensor, nudge_tensor


class Batch:
    """
    Batch is a data class consisting of fields of tensors like past_output_ids, past_key_values. It's a compact collection of variables that need to be updated in each incremental inference call.
    """

    def __init__(self,
                 next_input_ids: torch.Tensor = None,
                 past_key_values=None):
        # [batch, 1 or topk]
        self.next_input_ids = next_input_ids
        # [batch, heads, seq_past, kv_dim=42]
        self.past_key_values = past_key_values

        # Fields in children class
        self.top_k_probs = None
        self.past_output_ids = None
        self.beam_prob = None

    def merge(self, batch: Batch, seq_delta: int) -> Batch:
        """
        merges another batch with itself.
        """
        past_key_values = []
        for kv_pair1, kv_pair2 in zip(self.past_key_values,
                                      batch.past_key_values):
            kv = tuple()
            for kv1, kv2 in zip(kv_pair1, kv_pair2):
                kv += (merge_tensors(kv1,
                                     kv2,
                                     seq_delta=seq_delta,
                                     seq_order=2), )
            past_key_values.append(kv)
        self.past_key_values = tuple(past_key_values)

        self.next_input_ids = merge_tensors(self.next_input_ids,
                                            batch.next_input_ids,
                                            seq_delta=seq_delta,
                                            seq_order=-1)
        return self

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):
        past_key_values = []
        for k, v in self.past_key_values:
            k = trim_tensor(k,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=2)
            v = trim_tensor(v,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=2)
            past_key_values.append((k, v))
        self.past_key_values = tuple(past_key_values)

        self.next_input_ids = trim_tensor(self.next_input_ids,
                                          keep_indices=keep_indices,
                                          trim_seq_len=trim_seq_len,
                                          seq_order=-1)

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor,
                                        init_kv_cache_len: int):
        """
        This is used with a prefix kv_cache input. The init_seq_len part of the tensor is nudged towards right,
        by the displacement specified in offsets, so as to squeeze the padding bubble.
        """
        past_key_values = []
        for k, v in self.past_key_values:
            past_key_values.append((nudge_tensor(k,
                                                 offsets,
                                                 init_kv_cache_len,
                                                 seq_order=2),
                                    nudge_tensor(v,
                                                 offsets,
                                                 init_kv_cache_len,
                                                 seq_order=2)))
        self.past_key_values = tuple(past_key_values)

        # The past_hidden_states doesn't need to nudge, since the prefix kv_cache is also padded hidden_states


class ContrastiveBatch(Batch):

    def __init__(self,
                 next_input_ids: torch.tensor = None,
                 past_key_values=None,
                 past_output_ids: torch.tensor = None,
                 top_k_probs: torch.tensor = None):
        super().__init__(past_key_values=past_key_values,
                         next_input_ids=next_input_ids)  # [batch, topk]
        # [batch, past_seq]
        self.past_output_ids = past_output_ids
        # [batch, topk]
        self.top_k_probs: torch.Tensor = top_k_probs

    # merges another batch with itself.
    def merge(self, batch: ContrastiveBatch,
              seq_delta: int) -> ContrastiveBatch:
        self.past_output_ids = merge_tensors(self.past_output_ids,
                                             batch.past_output_ids,
                                             seq_delta=seq_delta,
                                             seq_order=1)
        self.top_k_probs = merge_tensors(self.top_k_probs,
                                         batch.top_k_probs,
                                         seq_delta=seq_delta,
                                         seq_order=-1)
        super().merge(batch, seq_delta)
        return self

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):
        self.past_output_ids = trim_tensor(self.past_output_ids,
                                           keep_indices=keep_indices,
                                           trim_seq_len=trim_seq_len,
                                           seq_order=1)
        self.top_k_probs = trim_tensor(self.top_k_probs,
                                       keep_indices=keep_indices,
                                       trim_seq_len=trim_seq_len,
                                       seq_order=-1)
        super().trim(keep_indices, trim_seq_len)

    def nudge_to_squeeze_bubble_padding(self, offsets: torch.Tensor,
                                        init_kv_cache_len: int):
        self.past_output_ids = nudge_tensor(self.past_output_ids,
                                            offsets,
                                            init_kv_cache_len,
                                            seq_order=1)
        super().nudge_to_squeeze_bubble_padding(offsets, init_kv_cache_len)


class BeamBatch(Batch):

    def __init__(self, past_key_values, beam_prob: torch.Tensor,
                 past_output_ids: torch.Tensor):
        super().__init__()
        # [batch, beam]
        self.beam_prob = beam_prob
        # [batch, beam, past_seq]
        self.past_output_ids = past_output_ids
        # past_key_values: [batch, beam, heads, seq_past, kv_dim]
        self.past_key_values = past_key_values

    # merges another batch with itself.
    def merge(self, batch: BeamBatch, seq_delta: int) -> BeamBatch:
        self.beam_prob = merge_tensors(self.beam_prob,
                                       batch.beam_prob,
                                       seq_delta=seq_delta,
                                       seq_order=-1)

        self.past_output_ids = merge_tensors(self.past_output_ids,
                                             batch.beam_prob,
                                             seq_delta=seq_delta,
                                             seq_order=2)

        past_key_values = []
        for kv_pair1, kv_pair2 in zip(self.past_key_values,
                                      batch.past_key_values):
            kv = tuple()
            for kv1, kv2 in zip(kv_pair1, kv_pair2):
                kv += (merge_tensors(kv1,
                                     kv2,
                                     seq_delta=seq_delta,
                                     seq_order=3), )
            past_key_values.append(kv)
        self.past_key_values = tuple(past_key_values)

        return self

    def trim(self, keep_indices: torch.Tensor, trim_seq_len: int):
        self.beam_prob = trim_tensor(self.beam_prob,
                                     keep_indices=keep_indices,
                                     trim_seq_len=trim_seq_len,
                                     seq_order=-1)

        self.past_output_ids = trim_tensor(self.past_output_ids,
                                           keep_indices=keep_indices,
                                           trim_seq_len=trim_seq_len,
                                           seq_order=2)
        past_key_values = []
        for k, v in self.past_key_values:
            k = trim_tensor(k,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=3)
            v = trim_tensor(v,
                            keep_indices=keep_indices,
                            trim_seq_len=trim_seq_len,
                            seq_order=3)
            past_key_values.append((k, v))
        self.past_key_values = tuple(past_key_values)
