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
        self.embedder = None

    @abstractmethod
    def forward(
        self, inputs: List[torch.tensor], past_key_values: Union[Tuple, None]
    ) -> Tuple[torch.tensor, Tuple, torch.tensor]:
        """
        Convert the variables between that used in the internal model's forward call and that used in the
        autoregressive search.

        Args:
            inputs (`List[torch.tensor]`):
                [input_ids, position_ids, attention_mask], order preserved.
                `input_ids`: [batch_size, input_seq_len]
                `position_ids`: [batch_size, input_seq_len],
                `attention_mask`: [batch_size, past_seq_len + input_seq_len].
            past_key_values (`Tuple`):
                The kv_cache. The required form of kv_cache used in the autoregressive search is Tuple[Tuple[key,
                value] * num_layers]  TODO: It should be serialized to List[torch.tensor]
                key: (batch_size, num_heads, seq_len, kv_dim),
                value: (batch_size, num_heads, seq_len, kv_dim).

        Returns:
            logits (`torch.tensor`):
                (batch_size, vocab_dim)
            past_key_values (`Tuple`):
                same as above.
            first_layer_hidden_state ('torch.tensor`):
                (batch_size, seq_len, hidden_dim), the embedding of the tokens.
        """
        pass

    def get_embedder(self):
        try:
            self.embedder = self.model.get_input_embeddings()
        except Exception:
            from functools import wraps
            from collections import OrderedDict

            # Create a customized LRU cache to get the embedding of past_seq_len part of input_ids tensor
            def lru_cache_part_tensor(maxsize=None):

                def decorator(embed_func):
                    lru_cache = OrderedDict()

                    @wraps(embed_func)
                    def wrapper(input_ids):
                        if len(input_ids.shape) >= 2:
                            # Convert the slice to a tuple as the cache key
                            key = tuple(input_ids[:, :-1].flatten().tolist())
                            # Last column computed without using cache
                            tensor_last_col = input_ids[:, -1]

                            # Check if the key is already in the cache
                            if key and key in lru_cache:
                                result = torch.cat([
                                    lru_cache[key],
                                    embed_func(tensor_last_col)
                                ],
                                                   dim=1)
                                new_key = tuple(input_ids.flatten().tolist())
                                if maxsize is not None and len(
                                        lru_cache) + 1 > maxsize:
                                    # If cache size exceeds the maximum, remove by FIFO order
                                    lru_cache.popitem(last=False)
                                    lru_cache[new_key] = result
                                return result

                        # If not in cache, compute the result and store it in the cache
                        key = tuple(input_ids.flatten().tolist())
                        result = embed_func(input_ids)
                        if maxsize is not None and len(
                                lru_cache) + 1 > maxsize:
                            # If cache size exceeds the maximum, remove by FIFO order
                            lru_cache.popitem(last=False)
                            lru_cache[key] = result

                        return result

                    return wrapper

                return decorator

            @lru_cache_part_tensor(3)
            def hidden_state_embedding(input_ids: torch.tensor):
                # input_ids: [batch, seq_len]
                position_ids = torch.zeros_like(input_ids)
                attention_mask = torch.ones_like(input_ids)
                _, _, hidden_states = self.forward(
                    [input_ids, position_ids, attention_mask], None)
                # [input_ids.shape, hidden_dim]
                return hidden_states[
                    0]  # Take the first layer hidden_states as token embedding

            self.embedder = hidden_state_embedding

    def embedding(self, input_ids: torch.tensor):
        """
        Get the embedding of input_ids. This is used only in contrastive search.
        Users can choose one of the following three ways to provide an embedder:
        1. make sure self.model.get_input_embedding() works;
        2. self.model.forward() allows output_hidden_states;
        3. provide an embedder at instantiation.

        Args:
            input_ids (`torch.tensor`):
                [batch, seq_len]

        Returns:
            (`torch.tensor`): [batch, seq_len, hidden_dim]
        """
        if self.embedder is None:
            try:
                self.get_embedder()
            except Exception as e:
                raise (
                    e,
                    Exception(
                        "Contrastive search requires an embedder but no working embedder is found. Please choose one of "
                        "the following three ways to provide an embedder: 1. self.model.get_input_embedding() works;"
                        "2. self.model.forward() allows output_hidden_states; 3. provide an embedder at instantiation."
                    ))

        # [input_ids.shape, hidden_dim]
        return self.embedder(input_ids).view(input_ids.shape + (-1, ))


class HuggingfaceBlock(LMBlock):

    def __init__(self, *args):
        super(HuggingfaceBlock, self).__init__(*args)
        self.config = {
            'use_cache': True,
            'return_dict': False,
            'output_attentions': False,
            'output_hidden_states': False
        }

    def forward(self, inputs: List[torch.tensor],
                past_key_values: Union[Tuple, None]):
        # Pre-process
        inputs = [input.contiguous() for input in inputs]
        if past_key_values is not None:
            new_kv_list = []
            for k, v in past_key_values:
                k_new = k.contiguous()
                v_new = v.contiguous()
                new_kv_list.append((k_new, v_new))
            past_key_values = tuple(new_kv_list)

        # Forward
        logits, past_key_values = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            **self.config)

        # Post-process
        return logits, past_key_values


class BloomBlock(LMBlock):

    def __init__(self, *args):
        super(BloomBlock, self).__init__(*args)
        self.config = {
            'use_cache': True,
            'return_dict': False,
            'output_attentions': False,
            'output_hidden_states': False
        }

    def forward(self, inputs: List[torch.tensor], past_key_values):
        # inputs: [input_ids, position_ids, attention_mask]
        # kv: (batch, num_head, seq_len, kv_dim) <->
        # k: (batch*num_head, kv_dim, seq_len), v: (batch*num_head, seq_len, kv_dim)
        batch_size = inputs[0].shape[0]

        # Pre-process
        if past_key_values is not None:
            _, num_head, seq_len, kv_dim = past_key_values[0][0].shape
            new_kv_list = []
            for k, v in past_key_values:
                k_new = torch.permute(
                    k.view(batch_size * num_head, seq_len, kv_dim),
                    (0, 2, 1)).contiguous()
                v_new = v.view(batch_size * num_head, seq_len,
                               kv_dim).contiguous()
                new_kv_list.append((k_new, v_new))
            past_key_values = tuple(new_kv_list)

        inputs = [input.contiguous() for input in inputs]

        # Forward
        logits, past_key_values = self.model.forward(
            input_ids=inputs[0],
            position_ids=inputs[1],
            attention_mask=inputs[2],
            past_key_values=past_key_values,
            **self.config)

        # Post-process
        _, kv_dim, seq_len = past_key_values[0][0].shape
        new_kv_list = []
        for k, v in past_key_values:
            k_new = torch.permute(k, (0, 2, 1)).view(batch_size, -1, seq_len,
                                                     kv_dim)
            v_new = v.view(batch_size, -1, seq_len, kv_dim)
            new_kv_list.append((k_new, v_new))
        past_key_values = tuple(new_kv_list)

        return logits, past_key_values
