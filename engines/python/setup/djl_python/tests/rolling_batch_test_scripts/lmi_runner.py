import torch
import os
import sys

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../../setup"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
from djl_python.tests.rolling_batch_test_scripts.generator import Generator, print_rank0

from benchmark_utils import timeit, parse_input, PeakMemory


class RunnerLmi:

    def __init__(self, model_id, device, param: dict, properties: dict):
        device = int(os.environ.get("RANK", 0))
        properties["device"] = int(os.environ.get("RANK", 0))

        from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch

        rolling_batch = LmiDistRollingBatch(model_id, device, properties)
        rolling_batch.output_formatter = None

        self.gen = Generator(rolling_batch=rolling_batch)
        self.param = param

    @torch.no_grad()
    def pure_inference(self, request_uids, input_str, max_infer_call=-1):
        """
        Add requests and run to the end
        """

        self.gen.reset()
        peak_memory = PeakMemory()
        torch.cuda.reset_max_memory_allocated()        

        # Add              
        N = len(input_str)
        params = [self.param.copy() for _ in range(N)]

        self.gen.step(step=0, input_str_delta=input_str, params_delta=params)

        # Run inference
        cnt = 0
        while True:
            self.gen.step(step=1)

            peak_memory.aggregate()

            if max_infer_call >= 0:
                print_rank0(f"Progress: {cnt/max_infer_call}_________\n")
            cnt += 1

            if self.gen.is_empty():
                break

        return self.gen.output_all, self.gen.token_numbers, 0, torch.cuda.max_memory_allocated() / 1024**2, peak_memory.get() / 1024**2

    def release_cache(self):
        self.gen.rolling_batch.release_cache()
