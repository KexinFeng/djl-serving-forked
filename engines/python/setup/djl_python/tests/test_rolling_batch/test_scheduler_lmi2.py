from collections import defaultdict
import torch
from djl_python.rolling_batch import LmiDistRollingBatch, SchedulerRollingBatch
import torch.distributed as dist

def print_rank0(content):
    rank = 0
    if dist.is_initialized():
        rank = dist.get_rank()
    if rank == 0:
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

model_id = "TheBloke/Llama-2-13B-Chat-fp16"
# model_id = "openlm-research/open_llama_7b_v2"
# model_id = "huggyllama/llama-7b"

"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":1000, "do_sample":true, "temperature":0.7}}
"""
input_str = [r"write a program to add two numbers in python", 
             r"Deep mind is a",
             r"Memories follow me left and right. I can"]

params = [{"max_new_tokens":50, "do_sample":False, "temperature":0.001}, 
          {"max_new_tokens":50, "do_sample":False, "temperature":0.001},
          {"max_new_tokens":50, "do_sample":False, "temperature":0.001}]

# ===================== lmi ============================
print("=========== before =========")
# rolling_batch = SchedulerRollingBatch(model_id, device, properties)
rolling_batch = LmiDistRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None
print("reach here")

output_all = defaultdict(list)
result = rolling_batch.inference(input_str, params)
for i, res in enumerate(result):
    output_all[i].append(res['data']) 
    
for _ in range(50):
    result = rolling_batch.inference([], [])
    for i, res in enumerate(result):
        output_all[i].append(res['data']) 

for i, out in enumerate(output_all.values()):
    print_rank0(input_str[i] + ''.join(out))
    print_rank0('\n====')


"""
write a program to add two numbers in python

Here is a simple program to add two numbers in Python:
```
a = 5
b = 3

sum = a + b

print("The sum is:", sum)
```
This program will

====
Deep mind is a term used to describe the collective unconscious of humanity, a shared reservoir of archetypes and memories that underlie our individual experiences and dreams. It is a concept developed by Carl Jung, a Swiss psychiatrist

====
Memories follow me left and right. I can't seem to shake them off. They are like a shadow that haunts me wherever I go.

I try to distract myself with work, with hobbies, with anything that can take my mind off of them.

====
"""