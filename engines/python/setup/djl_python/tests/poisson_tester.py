from typing import List, Tuple, Union
import bisect

import argparse

import numpy as np


class IterationScheduler:
    def __init__(self):
        pass


class RequestScheduler:
    def __init__(self):
        pass


class BatchRequest:
    def __init__(self):
        # Individual occurance
        self.uids = None
        self.time_queues = None  # Valid time_ques excluding the last one which is leftover
        # self.init_seq_lens = None
        self.input_ids: List[List[int]] = []

        # Whole batch
        self.batch_size = None
        self.time_batch_start = None
        self.time_batch_end = None

    def _to_list(self):
        return [self.uids, self.time_queues, self.input_ids]

    def from_list(self, data_list):
        self.uids, self.time_queues, self.input_ids = data_list

    def trim(self, max_batch_size):
        if self.batch_size <= max_batch_size:
            return None

        wait_request = BatchRequest()
        keep_list = []
        wait_list = []
        for data in self._to_list():
            keep_list.append(data[:max_batch_size])
            wait_list.append(data[max_batch_size:])
        self.from_list(keep_list)
        wait_request.from_list(wait_list)

        wait_request.batch_size = self.batch_size - max_batch_size
        self.batch_size = max_batch_size

        return wait_request


class RequestGen:
    """
    First dry run with sleep(simulated_execution_time) to get the overhead purely from RequestGen. Then it is
    subtracted from the actual execution of the object scheduler.
    """
    def __init__(self, avg_occur_per_time, init_seqlen_intv, vocab_size, max_batch_size):
        # parameters
        self.avg_occur_per_time = avg_occur_per_time
        self.init_seqlen_intv = init_seqlen_intv
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size

        # state
        self.left_over: Union[BatchRequest, None] = None
        self.last_time_que = None
        self.uid = 0

    def generate(self, time_stamp) -> BatchRequest:
        time_span = time_stamp - self.last_time_que if self.last_time_que else 0
        delta_time_ques, cnt = self.poisson_gen(self.avg_occur_per_time, time_span)

        time_ques = delta_time_ques + self.last_time_que if self.last_time_que else 0
        self.last_time_que = time_ques[-1]

        event = BatchRequest()
        event.time_queues = time_ques[:-1]
        batch_size = len(event.time_queues)
        assert batch_size == cnt - 1

        event.uids = self.uid + np.arange(batch_size)
        self.uid += batch_size
        self.batch_size = batch_size

        # Input_size is randomly generated
        init_seq_lens = np.random.uniform(*self.init_seqlen_intv, size=batch_size)

        # Generate input_ids
        event.input_ids = [np.random.choice(self.vocab_size, seq_len) for seq_len in init_seq_lens]

        # Merge into the left_overs from the latest (below). Left_overs won't affect last_time_que
        self.left_over.merge(event)

        # Trim. Trim cuts the above for processing, keeps the below as left_over. This won't affect the last_time_que
        wait_request = self.left_over.trim(max_batch_size=self.max_batch_size)
        to_process = self.left_over
        self.left_over = wait_request
        return to_process

    @staticmethod
    def poisson_gen(avg_occur_per_time, time_span) -> Tuple[np.array, int]:
        """
        Returns the occurrence time stamps and number of ocurrences
        """
        mu = time_span * avg_occur_per_time
        sig = np.sqrt(mu)

        random_array = np.random.rand(int(mu + 4 * sig))
        occur_time_stamps = np.cumsum(-1 / avg_occur_per_time * np.log(random_array))
        idx = bisect.bisect_right(occur_time_stamps, time_span)
        return np.concatenate([0, occur_time_stamps[:idx + 1]]), idx + 2


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
