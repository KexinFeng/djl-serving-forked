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
    def __init__(self, model, embedder=None, max_embed_lru=10):
        """
        Set self.model to the input language model.
        """
        self.model = model
        # Used only in contrastive search. Requiring model to expose get_input_embeddings is a less tight requirement
        # than requiring model to output_hidden_states.
        self.embedder = embedder

        # Max embedding lru_cache size is twice the number of ContrastiveSeqBatcher's to run in a
        # scheduler.inference_call(). Used only in contrastive search.
        self.max_embed_lru = max_embed_lru

    @abstractmethod
    def forward(
            self, inputs: List[torch.tensor],
            past_key_values: Union[Tuple, None]) -> Tuple[torch.tensor, Tuple]:
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
        if hasattr(self.model, 'get_input_embeddings'):
            self.embedder = self.model.get_input_embeddings()
        else:
            warnings.warn(
                f"self.model.get_input_embeddings is not found. In following, the hidden_states is used as "
                f"embedding. But it is slow!")

            from functools import wraps
            from collections import OrderedDict

            def lru_cache_part_tensor(maxsize=None):
                """
                This is a customized LRU cache designed for input_ids: torch.tensor: [batch, seq_len], which is used
                to store the past_hidden_states that corresponds to input_ids[:, :-1]. More specifically, each time it
                receives input_ids, it uses input_ids[:, :-1] part as the key to look for the cache. Then,
                if found, use the cache to concatenates with the hidden_states of the input_ids[:, -1] part;
                if not compute the hidden_states of the whole input_ids. Finally use the whole input_ids as the key
                to store the result hidden_state.
                """

                def decorator(embed_func):
                    lru_cache = OrderedDict()

                    @wraps(embed_func)
                    def wrapper(input_ids):
                        if len(input_ids.shape) >= 2:
                            # Convert the slice to a tuple as the cache key
                            key = tuple(input_ids[:, :-1].flatten().tolist())
                            # Last column computed without using cache
                            tensor_last_col = input_ids[:, -1].view(-1, 1)

                            # Check if the key is already in the cache
                            if key and key in lru_cache:
                                result = torch.cat([
                                    lru_cache.pop(key),
                                    embed_func(tensor_last_col)
                                ],
                                                   dim=1)
                                new_key = tuple(input_ids.flatten().tolist())
                                lru_cache[new_key] = result
                                return result

                        # If not in cache, compute the result and store it in the cache
                        if maxsize is not None and len(
                                lru_cache) + 1 > maxsize:
                            # If cache size exceeds the maximum, remove by FIFO order
                            lru_cache.popitem(last=False)
                        key = tuple(input_ids.flatten().tolist())
                        result = embed_func(input_ids)
                        lru_cache[key] = result

                        return result

                    return wrapper

                return decorator

            @lru_cache_part_tensor(self.max_embed_lru)
            def hidden_state_embedding(input_ids: torch.tensor):
                # input_ids: [batch, seq_len]
                position_ids = torch.zeros_like(input_ids)
                attention_mask = torch.ones_like(input_ids)
                hidden_states = self.model.forward(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    return_dict=True)['hidden_states']
                # [input_ids.shape, hidden_dim]
                return hidden_states[
                    0]  # Take the first layer hidden_states as token embedding

            self.embedder = hidden_state_embedding

    def embedding(self, input_ids: torch.tensor):
        """
        Get the embedding of input_ids. This is used only in contrastive search.
        Users can choose one of the following three ways to provide an embedder:
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

    def __init__(self, *args):
        super(HuggingfaceBlock, self).__init__(*args)
        self.config = {
            'use_cache': True,
            'return_dict': True,
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
        output = self.model.forward(input_ids=inputs[0],
                                    position_ids=inputs[1],
                                    attention_mask=inputs[2],
                                    past_key_values=past_key_values,
                                    **self.config)
        logits, past_key_values = output['logits'], output['past_key_values']

        # Post-process
        return logits, past_key_values


class BloomBlock(LMBlock):

    def __init__(self, *args):
        super(BloomBlock, self).__init__(*args)
        self.config = {
            'use_cache': True,
            'return_dict': True,
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
        output = self.model.forward(input_ids=inputs[0],
                                    position_ids=inputs[1],
                                    attention_mask=inputs[2],
                                    past_key_values=past_key_values,
                                    **self.config)
        logits, past_key_values = output['logits'], output['past_key_values']

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
