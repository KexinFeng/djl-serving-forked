from collections import defaultdict
import torch

import os, sys
script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
import torch.distributed as dist

def print_rank0(content):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(content)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""
option.model_id=TheBloke/Llama-2-13B-Chat-fp16
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.model_loading_timeout=7200
option.max_rolling_batch_prefill_tokens=36080
"""
properties = {"tensor_parallel_degree": 2,
              "dtype": "fp16",
              "max_rolling_batch_size": 8,
              "model_loading_timeout": 7200,
              "max_rolling_batch_prefill_tokens": 10000,
              "paged_attention": "True"}

# model_id = "TheBloke/Llama-2-13B-Chat-fp16"
# model_id = "openlm-research/open_llama_7b_v2"
model_id = "huggyllama/llama-7b"


"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":300, "do_sample":true, "temperature":0.001}}
"""

# ===================== Runner Class =====================
class Generator:
    def __init__(self, rolling_batch):
        self.rolling_batch = rolling_batch

        # Store the results
        self.output_all = defaultdict(list)
        self.input_all = {}

        # Status variables, the remaining
        self.input_str = []
        self.params = []
        self.req_ids = []
    
    def step(self, step=20, input_str_delta=None, params_delta=None):
        if input_str_delta:
            begin_id = max(self.input_all.keys(), default=0) + 1
            req_ids_delta = list(range(begin_id, begin_id + len(input_str_delta)))

            self.input_str += input_str_delta
            self.params += params_delta
            self.req_ids += req_ids_delta
            for req_id, input_s, param in zip(req_ids_delta, input_str_delta, params_delta):
                self.input_all[req_id] = (input_s, param) 

        for _ in range(step):
            result = rolling_batch.inference(self.input_str, self.params)
            for res, req_id in zip(result, self.req_ids):
                self.output_all[req_id].append(res['data'])
            self.req_ids = [req_id for req_id, res in zip(self.req_ids, result) if not res['last']]
            self.input_str = [s for s, res in zip(self.input_str, result) if not res['last']]
            self.params = [p for p, res in zip(self.params, result) if not res['last']]
            if not self.req_ids:
                break


# ===================== lmi ============================
print("=========== before =========")
# rolling_batch = SchedulerRollingBatch(model_id, device, properties)
rolling_batch = LmiDistRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None
print("reach here")

gen = Generator(rolling_batch=rolling_batch)

print('========== init inference ===========')
input_str1 = ["write a program to add two numbers in python",
              "write a program to add two numbers in python\n"]
params1 = [{"max_new_tokens":236, "do_sample":True, "temperature":0.001},
           {"max_new_tokens":236, "do_sample":True, "temperature":0.001}]

gen.step(input_str_delta=input_str1, params_delta=params1)

for _ in range(7):
    print('========== inference1 ===========')
    input_str_delta = ["write a program to add two numbers in python",
                "write a program to add two numbers in python\n"]

    params_delta = [{"max_new_tokens":236, "do_sample":True, "temperature":0.001},
            {"max_new_tokens":236, "do_sample":True, "temperature":0.001}]

    gen.step(input_str_delta=input_str_delta, params_delta=params_delta)


print('========== inference_infty ===========')
gen.step(step=500)
for req_id, out in gen.output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{gen.input_all[req_id][0] + ''.join(out)}\n")

