from typing import List

import torch
from utils import merge_tensors
from __future__ import annotations

from djl_python.scheduler.batch import Batch

PAD_TOKEN_ID = 220


class ContrastiveBatch(Batch):

    def __init__(
        self,
        seq_dim_order: List[int],
        past_output_ids: torch.Tensor = None,
        past_attention_mask: torch.Tensor = None,
        past_hidden_states: torch.Tensor = None,
        logits: torch.Tensor = None,
        past_key_values: List[torch.Tensor] = None,
    ):
        self.seq_dim_order = seq_dim_order
        self.past_output_ids = past_output_ids
        self.past_attention_mask = past_attention_mask
        self.past_hidden_states = past_hidden_states
        self.past_key_values = past_key_values
        self.logits = logits

    def merge(self, batch: ContrastiveBatch, seq_delta):
        past_output_ids = merge_tensors(self.past_output_ids,
                                        batch.past_output_ids,
                                        seq_delta=seq_delta,
                                        seq_order=self.seq_dim_order[0],
                                        is_pad_token=True)
        past_attention_mask = merge_tensors(self.past_attention_mask,
                                            batch.past_attention_mask,
                                            seq_delta=seq_delta,
                                            seq_order=self.seq_dim_order[1])

        past_hidden_states = merge_tensors(self.past_hidden_states,
                                           batch.past_hidden_states,
                                           seq_delta=seq_delta,
                                           seq_order=self.seq_dim_order[2])
        logits = merge_tensors(self.logits,
                               batch.logits,
                               seq_delta=seq_delta,
                               seq_order=self.seq_dim_order[3])

        past_key_values = []
        for i in range(len(self.past_key_values)):
            past_key_values.append(
                merge_tensors(self.past_key_values[i],
                              batch.past_key_values[i],
                              seq_delta=seq_delta,
                              seq_order=self.seq_dim_order[i]))

        return ContrastiveBatch(seq_dim_order=self.seq_dim_order,
                                past_output_ids=past_output_ids,
                                past_attention_mask=past_attention_mask,
                                past_hidden_states=past_hidden_states,
                                logits=logits,
                                past_key_values=past_key_values)

    @staticmethod
    def trim_tensor(trim_sequence: int, keep_indices: List[int], seq_dim_order,
                    tensor: torch.Tensor):
        pass

    def trim(self, trim_sequence: int, keep_indices: List[int]):
        pass
