from collections import defaultdict
from djl_python.rolling_batch import LmiDistRollingBatch
import torch

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
              "paged_attention": "False"}

model_id = "TheBloke/Llama-2-13B-Chat-fp16"
model_id = "openlm-research/open_llama_7b_v2"

print("=========== before =========")
rolling_batch = LmiDistRollingBatch(model_id, device, properties)

print("reach here")

"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":1000, "do_sample":true, "temperature":0.7}}
"""
input_str = ["write a program to add two numbers in python", 
             "Deep mind is a"]
params = [{"max_new_tokens":50, "do_sample":False, "temperature":0.7}, 
          {"max_new_tokens":50, "do_sample":False, "temperature":0.7}]

output_all = defaultdict(list)
result = rolling_batch.inference(input_str, params)
for i, res in enumerate(result):
    output_all[i].append(res['data']) 
for _ in range(50):
    result = rolling_batch.inference([], [])
    for i, res in enumerate(result):
        output_all[i].append(res['data']) 
    print(output_all)