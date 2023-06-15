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
import warnings
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

import torch


class LMBlock(ABC):

    @abstractmethod
    def __init__(self, model, embedder=None):
        """
        Set self.model to the input language model.
        """
        self.model = model

        # Used only in contrastive search. Requiring model to expose get_input_embeddings is a less tight requirement
        # than requiring model to output_hidden_states.
        self.embedder = embedder

    @abstractmethod
    def forward(
        self, inputs: List[torch.tensor], past_key_values: Union[Tuple, None]
    ) -> Tuple[torch.tensor, Tuple, torch.tensor]:
        """
        Convert the variables between that used in the internal model's forward call and that used in the
        autoregressive search.

        Args:
            inputs (`List[torch.tensor]`):
                Contains [input_ids, position_ids, attention_mask], order preserved.
                `input_ids` and `position_ids` are of size (batch_size, input_seq_len),
                `attention_mask` is of size (batch_size, past_seq_len + input_seq_len).
            past_key_values (`Tuple`):
                The kv_cache. The required form of kv_cache used in the autoregressive search is Tuple[Tuple[key,
                value] * num_layers].
                key: (batch_size, num_heads, seq_len, kv_dim),
                value: (batch_size, num_heads, seq_len, kv_dim).
        Return:
            logits (`torch.tensor`):
                (batch_size, vocab_dim)
            past_key_values (`Tuple`):
                same as above.
            hidden_state ('torch.tensor`):
                (batch_size, seq_len, hidden_dim), the embedding of the tokens.
        """
        pass

    def get_embedder(self):
        if hasattr(self.model, 'get_input_embeddings'):
            self.embedder = self.model.get_input_embeddings()
        else:
            raise Exception(
                f"self.model.get_input_embeddings is not found. In following, the hidden_states is used as "
                f"embedding. But it is slow!")


    def embedding(self, input_ids: torch.tensor):
        """
        Get the embedding of input_ids. This is used only in contrastive search.
        Users can choose one of the following three ways to provide an embedder
        (Assume that requiring model to expose get_input_embeddings is a less
        tight requirement than requiring model to output_hidden_states):
        1. make sure self.model.get_input_embedding() works.
        2. provide an embedder at instantiation.
        3. self.model.forward() allows output_hidden_states. But this is slow.

        Args:
            input_ids (`torch.tensor`):
                [batch, seq_len]

        Returns:
            (`torch.tensor`): [batch, seq_len, hidden_dim]
        """
        if self.embedder is None:
            self.get_embedder()

        # [input_ids.shape, hidden_dim]
        return self.embedder(input_ids).view(input_ids.shape + (-1, ))


class HuggingfaceBlock(LMBlock):

    def __init__(self, model):
        super(HuggingfaceBlock, self).__init__(model)
        self.config = {
            'use_cache': True,
            'return_dict': False,
            'output_attentions': False,
            'output_hidden_states': True
        }

    def forward(self, inputs: List[torch.tensor],
                past_key_values: Union[Tuple, None]):
        logits, past_key_values, hidden_states = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            **self.config)

        # post-process
        return logits, past_key_values, hidden_states[
            0]  # take the lowest hidden_states as token embedding


class BloomBlock(LMBlock):

    def __init__(self, model):
        super(BloomBlock, self).__init__(model)
        self.config = {
            'use_cache': True,
            'return_dict': False,
            'output_attentions': False,
            'output_hidden_states': True
        }

    def forward(self, inputs: List[torch.tensor], past_key_values):
        # kv: (batch, num_head, seq_len, kv_dim)
        # <->
        # k: (batch*num_head, kv_dim, seq_len),
        # v: (batch*num_head, seq_len, kv_dim)
        batch_size = inputs[0].shape[0]

        # pre-process
        if past_key_values is not None:
            _, num_head, seq_len, kv_dim = past_key_values[0][0].shape
            new_kv_list = []
            for k, v in past_key_values:
                k_new = torch.permute(
                    k.view(batch_size * num_head, seq_len, kv_dim), (0, 2, 1))
                v_new = v.view(batch_size * num_head, seq_len, kv_dim)
                new_kv_list.append((k_new, v_new))
            past_key_values = tuple(new_kv_list)

        # inference
        logits, past_key_values, hidden_states = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            **self.config)

        # post-process
        _, kv_dim, seq_len = past_key_values[0][0].shape
        new_kv_list = []
        for k, v in past_key_values:
            k_new = torch.permute(k, (0, 2, 1)).view(batch_size, -1, seq_len,
                                                     kv_dim)
            v_new = v.view(batch_size, -1, seq_len, kv_dim)
            new_kv_list.append((k_new, v_new))
        past_key_values = tuple(new_kv_list)

        return logits, past_key_values, hidden_states[
            0]  # take the lowest hidden_states as token embedding
