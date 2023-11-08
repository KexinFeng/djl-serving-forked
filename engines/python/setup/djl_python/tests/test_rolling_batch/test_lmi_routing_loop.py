import torch

import os, sys
script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
from djl_python.tests.test_rolling_batch.generator import Generator, print_rank0

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
properties = {"tensor_parallel_degree": 1,
              "dtype": "fp16",
              "max_rolling_batch_size": 8,
              "model_loading_timeout": 7200,
              "max_rolling_batch_prefill_tokens": 1000,
              "paged_attention": "True"}

# model_id = "TheBloke/Llama-2-13B-Chat-fp16"  # multi gpu
# model_id = "openlm-research/open_llama_7b_v2"
# model_id = "huggyllama/llama-7b"
model_id = "bigscience/bloom-560m"  # OOM on a single gpu and not sharded on multi gpu
# model_id = "gpt2"

"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":300, "do_sample":true, "temperature":0.001}}
"""

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

