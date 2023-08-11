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
from collections import defaultdict, OrderedDict
from sortedcontainers import SortedDict
from typing import Union, Tuple, List, Dict, Type, Any

import torch

from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.lm_block import LMBlock
from djl_python.scheduler.seq_batcher import SeqBatcher
from djl_python.scheduler.seq_batcher_impl import GreedySeqBatcher, ContrastiveSeqBatcher
from djl_python.scheduler.utils import compute_kv_cache, compute_position_ids, compute_offsets, trim_tensor

SEARCH_ALGORITHM_TO_CLASS = {
    "greedy": GreedySeqBatcher,
    "sampling": GreedySeqBatcher,
    "contrastive": ContrastiveSeqBatcher
}


class SeqBatchScheduler:
    """
    This is a scheduler that manages the SeqBatcher, providing API which allows for actions like addBatch,
    collectResults.
    """

    def __init__(self, lm_block: LMBlock, default_search_algorithm: str,
                 default_config: SearchConfig):
        self.default_search_configs = defaultdict(lambda: default_config)
        self.default_seq_batcher_cls = SEARCH_ALGORITHM_TO_CLASS[
            default_search_algorithm]
        self.lm_block = lm_block
        self.results: Dict[int, List[int]] = defaultdict(list)

        self.seq_batchers: Dict[
            Type[SeqBatcher]:List[SeqBatcher]] = defaultdict(list)

        # Runtime prompt caching
        self.lru_kv_cache = OrderedDict()
        self.lru_max_size = 10

        # Optimal seqBatcher partition
        # (length, request_uid) -> (seq_batcher_idx, sub_seq_batcher_idx)
        self.seq_lengths_sorted: Dict[Type[SeqBatcher]:SortedDict] = {}

    def add_request(self,
                    input_ids: torch.Tensor,
                    request_uids: torch.Tensor,
                    search_algorithm: str = None,
                    search_configs: List[SearchConfig] = None,
                    kv_cache: Union[Tuple, None] = None,
                    kv_cache_prompt_ids: Union[Dict[int, torch.tensor],
                                               None] = None):
        """
        Args: kv_cache_prompt_ids = {request_uid -> List[token_ids]}
        """

        # Find the requests that uses kv_cache_prompt_ids
        index_not_use_prompt = []
        search_configs_not_use_prompt = []
        if search_configs:
            for idx, search_config in enumerate(search_configs):
                if search_config.use_lru_kv_cache:
                    request_uid = request_uids[idx].item()
                    if request_uid not in kv_cache_prompt_ids:
                        raise Exception(
                            f"request_uids = {request_uid}: search_config says use_kv_cache_prompt, "
                            f"but the prompt_ids is not provided.")
                    prompt_ids_tensor = kv_cache_prompt_ids[request_uid]
                    key = tuple(prompt_ids_tensor.flatten().tolist())
                    # lru operations
                    if key not in self.lru_kv_cache:
                        if len(self.lru_kv_cache) + 1 > self.lru_max_size:
                            # If cache size exceeds the maximum, remove by FIFO order
                            self.lru_kv_cache.popitem(last=False)
                        kv_cache_tuple = compute_kv_cache(
                            input_ids=prompt_ids_tensor,
                            lm_block=self.lm_block,
                            search_configs=[search_config])
                        kv_cache_new = []
                        for k, v in kv_cache_tuple:
                            k_new = k.cpu()
                            v_new = v.cpu()
                            kv_cache_new.append((k_new, v_new))
                        self.lru_kv_cache[key] = tuple(kv_cache_new)
                        self.lru_kv_cache.move_to_end(key)

                        # _add_request
                        self._add_request(input_ids[idx].view(1, -1),
                                          request_uids[idx].view(1, -1),
                                          search_algorithm, [search_config],
                                          kv_cache=kv_cache_tuple)
                    else:
                        # _add_request
                        self._add_request(input_ids[idx].view(1, -1),
                                          request_uids[idx].view(1, -1),
                                          search_algorithm, [search_config],
                                          kv_cache=self.lru_kv_cache[key])
                        self.lru_kv_cache.move_to_end(key)
                else:
                    index_not_use_prompt.append(idx)
                    search_configs_not_use_prompt.append(search_config)
        else:
            index_not_use_prompt = list(range(input_ids.shape[0]))
            search_configs_not_use_prompt = None

        if index_not_use_prompt:
            index_not_use_prompt = torch.tensor(index_not_use_prompt)
            self._add_request(input_ids[index_not_use_prompt],
                              request_uids[index_not_use_prompt],
                              search_algorithm, search_configs_not_use_prompt,
                              kv_cache)

    def _add_request(self,
                     input_ids: torch.Tensor,
                     request_uids: torch.Tensor,
                     search_algorithm: str = None,
                     search_configs: List[SearchConfig] = None,
                     kv_cache: Union[Tuple, None] = None):
        # TODO: next, this will take an argument of `action`, computed by self.optimal_action.
        device = input_ids.device
        request_uids = request_uids.to(device)
        seq_batcher_cls = SEARCH_ALGORITHM_TO_CLASS.get(
            search_algorithm, self.default_seq_batcher_cls)
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

        seq_batcher_cls = self.default_seq_batcher_cls if seq_batcher_cls is None else seq_batcher_cls

        # Corner case: input_ids are empty. Pad them.
        if input_ids.numel() == 0:
            batch_size = input_ids.shape[0]
            input_ids = torch.zeros(batch_size,
                                    1,
                                    dtype=torch.int64,
                                    device=input_ids.device)
            for i in range(batch_size):
                input_ids[i, 0] = self.default_search_configs[
                    request_uids[i].item()].pad_token_id

        # Prefill
        new_seq_batcher, output_ids = seq_batcher_cls.init_forward(
            input_ids=input_ids,
            request_uids=request_uids,
            lm_block=self.lm_block,
            search_configs=self.default_search_configs,
            kv_cache=kv_cache)

        # Set the search_config._max_seqlen
        for idx, request in enumerate(request_uids.view(-1).tolist()):
            init_seqlen = len(
                input_ids[idx]) - new_seq_batcher.offsets[idx].item()
            if kv_cache:
                init_seqlen += kv_cache[0][0].shape[-2]
            # TODO: change search_configs dict to list
            new_seq_batcher.search_configs[
                request]._max_seqlen = new_seq_batcher.search_configs[
                    request].max_new_seqlen + init_seqlen

        # Merge
        # TODO: next, an optimal action needs to be first computed, according to which the merge is done.
        if not self.seq_batchers[seq_batcher_cls]:
            self.seq_batchers[seq_batcher_cls].append(new_seq_batcher)
        else:
            self.seq_batchers[seq_batcher_cls][0].add_batch(new_seq_batcher)

        # collect the input into result
        for request_uid, output_id in zip(
                request_uids.view(-1).tolist(), output_ids):
            self.results[request_uid] = output_id

        # Add the new coming request
        # Try larger num_part until infeasible
        # If num_part_opt is updated, do the consolidation

    def is_empty(self):
        return all(seq_batcher.is_empty()
                   for seq_batcher_list in self.seq_batchers.values()
                   for seq_batcher in seq_batcher_list)

    def total_seq_batcher_num(self):
        # This is provided to the consumers, used as part of the max_seq_batcher thresholding mechanism.
        return sum(
            len(seq_batcher_list)
            for seq_batcher_list in self.seq_batchers.values())

    def total_batch_size(self) -> Dict[Type[SeqBatcher], int]:
        # This is provided to the consumers, used as part of the max_batch_size thresholding mechanism.
        batch_size = {}
        for key, seq_batcher_list in self.seq_batchers.items():
            batch_size[key] = sum(seq_batcher.batch_size
                                  for seq_batcher in seq_batcher_list)
        return batch_size

    def optimal_action(self,
                       input_ids: torch.Tensor,
                       request_uids: torch.Tensor,
                       lm_block: LMBlock,
                       cls: Type[SeqBatcher] = None,
                       search_configs: defaultdict[Any, SearchConfig] = None,
                       kv_cache: Union[Tuple, None] = None):
        """
        Max_sparsity_threshold mechanism to save space.
            Case: 300 new token, pad 150 tokens, sparsity = 150/(150+300) = 0.33

        optimal number of partitions -> optimal partition

        Find the minimum number of partitions with the constraint that sparsity is below a threshold.
        Optimal_partition(num_part): dynamic programming O(num_part * batch_size)

        1. The optimal num_part may not necessarily change by one each time of input. But the object function
        monotonically depends on num_part. So just search by increasing the num_part by one and see if it is
        feasible.

        2. When to increase or decrease the num_part?
            a. Scenario of increasing num_part: _add_request() -> add_batch -> merge
                new_seq_batcher is by_default seen as an appending elem to the partition. Then run the optimization
                algorithm.
            b. Scenario of decreasing num_part: inference_call() -> collect_and_trim -> seq_batchers: Dict[int,
            List[int]].

        3. seq_batcher_partition optimization is an operation on self.seq_batchers: Dict[int, List[SeqBatcher]],
        thus in seq_batcher_scheduler class.

        4. Optimization search algorithm (see also point 1).
            a. Scenario of _add_request(), optimal num_part is only larger or equal. Search from num_part - 1 (
            new_seq_batcher is first appended) and upward, until feasible.
            b. Scenario of inference_call(), optimal num_part is only smaller or equal. Search from
            num_part - 1 and downward, until infeasible.

        Algo:
            Input: current_partitions, including the input_ids newly added, whose init_forward will be lazily
            called. This prevents the OOM at initialization, but sacrifices the init_forward efficiency (first token
            generation).

            Keep a sortedList of self.seq_lengths_sorted mapped to (src_partition_idx, src_sub_partition_idx). O(
            logN) insert and O(N) iteration for
                total_paddings, opt_partitions = optimal_partition(trial num_part).

            According total_paddings compute the sparsity and check. When the search is end, proceed to align the
            current_partitions to opt_partitions.

            Inside partition_align(opt_partition), it's going to operate on seq_batchers: Dict[
            int, List[SeqBatcher]], as well as the newly added input_ids, whose src_partition_idx = -1, which indicates
            an init_forward is needed.
                a. Collect the opt_partition into dict of src_partition_idx to get how each current_partition is
                split. I.e. target_partition = opt_partition = List of [idx_in_sortedList -> (src_partition_idx,
                sub_partition_idx)] => {src_partition_idx: List of [sub_partition_idx]}
                b. split and collect into target_partition_collector = {target_partition_idx: [seq_batchers]}.
                c. merge each seq_batchers in each target partition.
        """

        # seq_lengths_sorted: (length, request_uid) -> (seq_batcher_idx, sub_seq_batcher_idx)
        # seq_list: (length, seq_batcher_idx, sub_seq_batcher_idx)
        seq_list = list((key[0], val[0], val[1])
                        for key, val in self.seq_lengths_sorted[cls].items())
        total_tokens = sum(seq_list)

        # Loop and check sparsity < threshold
        trial_num_part = 3
        total_paddings, opt_partition = self.optimal_partition(
            [seq[0] for seq in seq_list], trial_num_part)
        sparsity = total_paddings / (total_paddings + total_tokens)

        # Accept the opt_partition, and align the self.seq_batchers
        num_part = len(opt_partition)
        # Build src_partition_collector, which is a tree that classifies the seq_list first by seq_batcher_idx then
        # by target_partition_idx.
        #   seq_batcher_idx -> target_partition_idx -> sub_seq_batcher_idx
        src_partition_collector = defaultdict(
            lambda: [list() for _ in range(num_part)])
        for target_partition_idx, partition in enumerate(opt_partition):
            for idx in partition:
                _, seq_batcher_idx, sub_seq_batcher_idx = seq_list[idx]
                # The seq_batcher_idx = -1 is reserved for newly added input_ids, whose init_forward is lazily called.
                src_partition_collector[seq_batcher_idx][
                    target_partition_idx].append(sub_seq_batcher_idx)

        # Split and collect into target_partition_collector, which is a tree:
        #   target_partition_idx -> list of seq_batchers
        target_partition_collector = defaultdict(list)
        for seq_batcher_idx, partitions in src_partition_collector.items():
            if seq_batcher_idx != -1:
                seq_batcher_split = self.seq_batchers[cls][
                    seq_batcher_idx].split(partitions)
                for idx, seq_batcher in enumerate(seq_batcher_split):
                    target_partition_collector[idx].append(seq_batcher)
            else:
                # The newly added input_ids calls init_forward() here.
                for idx, new_input_idx_subset in enumerate(partitions):
                    initial_offsets = compute_offsets(input_ids, [
                        search_configs[r].pad_token_id
                        for r in request_uids.view(-1).tolist()
                    ])
                    trim_len = min(initial_offsets[i]
                                   for i in new_input_idx_subset)
                    search_configs_subset = defaultdict(
                        search_configs.default_factory)
                    seq_batcher, _ = cls.init_forward(
                        trim_tensor(input_ids,
                                    new_input_idx_subset,
                                    trim_len,
                                    seq_order=1),
                        trim_tensor(request_uids,
                                    new_input_idx_subset,
                                    trim_len,
                                    seq_order=-1), lm_block,
                        search_configs_subset, kv_cache)
                    target_partition_collector[idx].append(seq_batcher)

        # Merge seq_batcher_list per target_partition and add to self.seq_batchers
        self.seq_batchers[cls] = [
            cls.add_batch_list(seq_batcher_list)
            for seq_batcher_list in target_partition_collector.values()
        ]

        # Maintain self.seq_lengths_sorted
        for seq_batcher_idx, seq_batcher in enumerate(self.seq_batchers[cls]):
            for idx, request_uid in enumerate(
                    seq_batcher.request_uids.view(-1).tolist()):
                length = seq_batcher.search_configs[request_uid]._max_seqlen
                self.seq_lengths_sorted[cls][length,
                                             request_uid] = (seq_batcher_idx,
                                                             idx)

    @staticmethod
    def optimal_partition(seq_length_list: List[int],
                          num_part: int) -> Tuple[int, List[List[int]]]:
        """
        total_padding, opt_partition = self.optimal_partition(
            seq_length_list, num_part)

        Args:
            seq_length_list: list of sequence lengths. Sorted in descending order.
            num_part: number of parts in a partition

        Return:
            cost: total padding
            opt_partition (`List[List[int]]`): optimal partition stored as List of List of sequence index
        """
        if num_part <= 0:
            raise Exception("Illegal argument.")

        batch_size = len(seq_length_list)
        arr = seq_length_list

        # dp[i][k] stores the optimal cost of partition the suffix array arr[i:] into k parts.
        dp = [[-1 for _ in range(num_part + 1)] for _ in range(batch_size)]

        # dp_parts[i][k] stores the corresponding optimal partition. dict: (i, k) -> List[int]
        dp_parts = defaultdict(list)

        def dp_recur(idx, k) -> Tuple[int, List[int]]:
            """
            dp(idx, k) returns the optimal cost of partition the suffix array arr[i:] into k parts.
            """
            if batch_size - idx < k:
                return 0 if idx == batch_size else float('inf'), []

            if dp[idx][k] > -1:
                return dp[idx][k], dp_parts[idx, k]

            if k == 1:
                max_seq_size = arr[idx]
                dp[idx][k], dp_parts[idx, k] = sum(
                    max_seq_size - arr[i]
                    for i in range(idx, batch_size)), [idx]
                return dp[idx][k], dp_parts[idx, k]

            if k == 2:

                def auc(i):
                    # arr[idx:] -> arr[idx: i], arr[i:]
                    return (batch_size - i) * (arr[idx] - arr[i])

                def optimal_bifurcate():
                    # Binary search for the peak point of the AUC: find the leftmost idx whose diff_auc <=0
                    diff_auc = lambda i: auc(i + 1) - auc(i)
                    lft, rgt = idx, batch_size - 2
                    while lft + 1 < rgt:
                        mid = lft + (rgt - lft) // 2
                        if diff_auc(mid) <= 0:
                            rgt = mid
                        else:
                            lft = mid + 1
                    cut = lft if diff_auc(lft) <= 0 else rgt
                    return cut

                opt_cut = optimal_bifurcate()
                dp[idx][2] = sum(
                    arr[idx] - arr[j]
                    for j in range(idx, batch_size)) - auc(opt_cut)
                dp_parts[idx, 2] = [opt_cut]
                return dp[idx][2], dp_parts[idx, 2]

            padding_leftmost_part = 0
            opt_cost, opt_cuts = float('inf'), None
            for i in range(idx, batch_size):
                padding_leftmost_part += arr[idx] - arr[i]
                padding_suffix_part, opt_cuts_suffix_part = dp_recur(
                    i + 1, k - 1)
                padding = padding_leftmost_part + padding_suffix_part
                if padding < opt_cost:
                    opt_cost = padding
                    opt_cuts = [i + 1] + opt_cuts_suffix_part

            dp[idx][k], dp_parts[idx, k] = opt_cost, opt_cuts
            return opt_cost, opt_cuts

        optimal_cost, optimal_cuts = dp_recur(0, num_part)

        # Convert the cuts to lists of sequence index divided in parts
        optimal_part = []
        for i in range(len(optimal_cuts)):
            optimal_part.append(
                list(
                    range(0 if i == 0 else optimal_cuts[i - 1],
                          optimal_cuts[i])))
        optimal_part.append(
            list(
                range(optimal_cuts[-1] if len(optimal_cuts) > 0 else 0,
                      batch_size)))
        return optimal_cost, optimal_part

    def inference_call(self) -> Tuple[List[List[int]], List[int], List[int]]:
        """
        A sweep of inference calls on all seq_batchers in the scheduler
        Returns:
            output_ids (`List[List[int]`):
                About List[List[int]] structure, the outermost List[] corresponds to request_uid: List[int]. The
                inner List[int] is used to extend the past_output_ids: past_output_ids.extend(List[
                int]). This is the same form as the output from `add_request`.
            request_uids (`List[int]`):
                The request_uids that correspond to output_ids. Ordering may be different from input since
                batch_merge or batch_trim operation.
            exist_request_uids (`List[int]`):
                List[int] a list of request_uids that have finished.
        """

        output: List[List[int]] = []
        request_uids: List[int] = []
        exit_request_uids: List[int] = []
        for seq_batcher_cls in self.seq_batchers:
            seq_batcher_list_new = []
            for seq_batcher in self.seq_batchers[seq_batcher_cls]:
                output += seq_batcher.forward()
                request_uids += seq_batcher.request_uids.view(-1).tolist()

                exit_request_uids += seq_batcher.collect_and_trim()
                if not seq_batcher.is_empty():
                    seq_batcher_list_new.append(seq_batcher)

            self.seq_batchers[seq_batcher_cls] = seq_batcher_list_new

            # Try smaller num_part until infeasible
            # If num_part_opt is updated, do the consolidation

        return output, request_uids, exit_request_uids

    def increment_forward(self, count: int):
        # This serves as a demo of how to use this scheduler
        # -> Dict[Type[SeqBatcher]: List[List[int]]]
        i = 0
        while i < count and not self.is_empty():
            output_ids, request_uids, _ = self.inference_call()

            # collect output
            for request_uid, output_id in zip(request_uids, output_ids):
                self.results[request_uid].extend(output_id)

            i += 1
            yield output_ids

    def collect_results(self):
        output = self.results
        self.results = defaultdict(list)
        return output

    def seq_batcher_split(self, seq_batcher_cls: Type[SeqBatcher],
                          seq_batcher_idx: int, partitions: List[List[int]]):
        """
        Split a seq_batcher in the seq_batcher_list located at seq_batcher_idx, into parts according to `partition`.
        Args:
            seq_batcher_cls: SeqBatcher type
            seq_batcher_idx: idx in the seq_batcher_list
            partitions: contains the seq_batcher_idx partitioned into lists.
        """

        seq_batcher = self.seq_batchers[seq_batcher_cls].pop(seq_batcher_idx)
        self.seq_batchers[seq_batcher_cls].extend(
            seq_batcher.split(partitions))

    def get_request_ids(self):
        request_uids = []
        for seq_batcher_cls in self.seq_batchers:
            for seq_batcher in self.seq_batchers[seq_batcher_cls]:
                request_uids += seq_batcher.request_uids.view(-1).tolist()

        return request_uids
