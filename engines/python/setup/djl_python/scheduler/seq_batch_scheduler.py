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
from collections import defaultdict
from typing import Union, Tuple, List

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.lm_block import LMBlock


class SeqBatchScheduler:

    def __init__(self, default_lm_block: LMBlock, default_seq_batcher_cls,
                 default_config: SearchConfig):
        self.default_search_configs = defaultdict(lambda: default_config)
        self.default_seq_batcher_cls = default_seq_batcher_cls
        self.default_lm_block = default_lm_block
        self.results = {}

        self.seq_batcher = None  # {key: seq_batcher}

    def add_request(self,
                    input_ids: torch.Tensor,
                    request_uids: torch.Tensor,
                    lm_block: LMBlock = None,
                    seq_batcher_cls=None,
                    search_configs: List[SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None,
                    save_kv_cache_path: str = None):
        device = self.default_lm_block.model.device
        request_uids = request_uids.to(device)
        input_ids = input_ids.to(device)
        if kv_cache:
            kv_list = []
            for k, v in kv_cache:
                k_new = k.to(device)
                v_new = v.to(device)
                kv_list.append((k_new, v_new))
            kv_cache = tuple(kv_list)

        if search_configs:
            for request, search_config in zip(
                    request_uids.view(-1).tolist(), search_configs):
                self.default_search_configs[request] = search_config

        lm_block_seq = self.default_lm_block if lm_block is None else lm_block
        seq_batcher_cls = self.default_seq_batcher_cls if seq_batcher_cls is None else seq_batcher_cls

        # prefill
        new_seq_batcher, output_ids = seq_batcher_cls.init_forward(
            input_ids=input_ids,
            request_uids=request_uids,
            lm_block=lm_block_seq,
            search_configs=self.default_search_configs,
            kv_cache=kv_cache,
            save_kv_cache_path=save_kv_cache_path)

        # merge
        if self.seq_batcher and not self.seq_batcher.is_empty():
            self.seq_batcher.add_batch(new_seq_batcher)
        else:
            self.seq_batcher = new_seq_batcher

        # collect the input into result
        for request_uid, output_id in zip(
                request_uids.view(-1).tolist(), output_ids):
            self.results[request_uid] = output_id

    def is_empty(self):
        return self.seq_batcher is None or self.seq_batcher.is_empty()

    def increment_forward(self, count: int) -> List[List[int]]:
        i = 0
        while i < count and not self.is_empty():
            # inference call
            output_ids = self.seq_batcher.inference_call()

            # collect output
            for request_uid, output_id in zip(
                    self.seq_batcher.request_uids.view(-1).tolist(),
                    output_ids):
                self.results[request_uid].extend(output_id)

            # trim the sequence batcher
            self.seq_batcher.collect_and_trim()

            i += 1
            yield output_ids

    def collect_results(self):
        output = self.results
        self.results = {}
        return output
