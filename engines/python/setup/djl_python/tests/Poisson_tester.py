import bisect

from djl_python.scheduler import HuggingfaceBlock, BloomBlock
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import AutoConfig
import torch
from collections import defaultdict

from transformers import AutoTokenizer, BloomForCausalLM

from djl_python.scheduler import SearchConfig
from djl_python.scheduler.seq_batcher_impl import GreedySeqBatcher, ContrastiveSeqBatcher
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from typing import List, Tuple
import bisect

import argparse

import numpy as np

class IterationScheduler:
    def __init__(self):
        pass

class RequestScheduler:
    def __init__(self):
        pass

def poisson_gen(avg_occur_per_time, time_span) -> Tuple[List[float], int]:
    """
    Returns the occurrence time stamps and number of ocurrences
    """
    mu = time_span * avg_occur_per_time
    sig = np.sqrt(mu)

    random_array = np.random.rand(int(mu + 4 * sig))
    occur_time_stamps = np.cumsum(-1/avg_occur_per_time*np.log(random_array))
    idx = bisect.bisect_right(occur_time_stamps, time_span)
    return occur_time_stamps[:idx], idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')

    parser.add_argument('-r', '--reps', dest='reps', type=int, default=1)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=2)
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560'],
                        default="bloom560")
    parser.add_argument('--batch_type',
                        type=str,
                        choices=['greedy', 'contrastive'],
                        default="greedy")
    args = parser.parse_args()
    for c in {1, 2, 4}:
        args.concurrency = c
        main(args)
