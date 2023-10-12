from collections import defaultdict
import torch
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

model_id = "TheBloke/Llama-2-13B-Chat-fp16"
# model_id = "openlm-research/open_llama_7b_v2"
# model_id = "huggyllama/llama-7b"

"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":300, "do_sample":true, "temperature":0.001}}
"""


# ===================== lmi ============================
print("=========== before =========")
# rolling_batch = SchedulerRollingBatch(model_id, device, properties)
rolling_batch = LmiDistRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None
print("reach here")

# Store the results
output_all = defaultdict(list)
input_all = {}

# Add request
input_str1 = ["write a program to add two numbers in python",
              "write a program to add two numbers in python\n"]
params1 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001},
           {"max_new_tokens":256, "do_sample":True, "temperature":0.001}]
req_ids1 = list(range(len(input_str1)))

for req_id, input_s, param in zip(req_ids1, input_str1, params1):
    input_all[req_id] = (input_s, param) 
input_str, params, req_ids = input_str1, params1, req_ids1

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference1 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


# Add request again
input_str2 = ["write a program to add two numbers in python"]

params2 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids2 = list(range(begin_id, begin_id + len(input_str2)))

input_str += input_str2
params += params2
req_ids += req_ids2
for req_id, input_s, param in zip(req_ids2, input_str2, params2):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference2 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


# Again for the 3rd time
input_str3 = ["write a program to add two numbers in python"]

params3 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids3 = list(range(begin_id, begin_id + len(input_str3)))

input_str += input_str3
params += params3
req_ids += req_ids3
for req_id, input_s, param in zip(req_ids3, input_str3, params3):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference3 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")



# Again for the 4th time
input_str4 = ["write a program to add two numbers in python"]

params4 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids4 = list(range(begin_id, begin_id + len(input_str4)))

input_str += input_str4
params += params4
req_ids += req_ids4
for req_id, input_s, param in zip(req_ids4, input_str4, params4):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference4 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


# Again for the 5th time
input_str5 = ["write a program to add two numbers in python"]

params5 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids5 = list(range(begin_id, begin_id + len(input_str5)))

input_str += input_str5
params += params5
req_ids += req_ids5
for req_id, input_s, param in zip(req_ids5, input_str5, params5):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference5 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


# Again for the 6th time
input_str6 = ["write a program to add two numbers in python"]

params6 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids6 = list(range(begin_id, begin_id + len(input_str6)))

input_str += input_str6
params += params6
req_ids += req_ids6
for req_id, input_s, param in zip(req_ids6, input_str6, params6):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference6 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")

# Again for the 7th time
input_str7 = ["write a program to add two numbers in python"]

params7 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids7 = list(range(begin_id, begin_id + len(input_str7)))

input_str += input_str7
params += params7
req_ids += req_ids7
for req_id, input_s, param in zip(req_ids7, input_str7, params7):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference7 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")

# Again for the 7th time
input_str7 = ["write a program to add two numbers in python"]

params7 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids7 = list(range(begin_id, begin_id + len(input_str7)))

input_str += input_str7
params += params7
req_ids += req_ids7
for req_id, input_s, param in zip(req_ids7, input_str7, params7):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference8 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")

# Again for the 7th time
input_str7 = ["write a program to add two numbers in python"]

params7 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids7 = list(range(begin_id, begin_id + len(input_str7)))

input_str += input_str7
params += params7
req_ids += req_ids7
for req_id, input_s, param in zip(req_ids7, input_str7, params7):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference9 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")

# Again for the 7th time
input_str7 = ["write a program to add two numbers in python"]

params7 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids7 = list(range(begin_id, begin_id + len(input_str7)))

input_str += input_str7
params += params7
req_ids += req_ids7
for req_id, input_s, param in zip(req_ids7, input_str7, params7):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference10 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")

# Again for the 7th time
input_str7 = ["write a program to add two numbers in python"]

params7 = [{"max_new_tokens":256, "do_sample":True, "temperature":0.001}]

begin_id = max(input_all.keys()) + 1
req_ids7 = list(range(begin_id, begin_id + len(input_str7)))

input_str += input_str7
params += params7
req_ids += req_ids7
for req_id, input_s, param in zip(req_ids7, input_str7, params7):
    input_all[req_id] = (input_s, param) 

for _ in range(1):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference11 ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


## The final
for _ in range(600):
    result = rolling_batch.inference(input_str, params)
    for res, req_id in zip(result, req_ids):
        output_all[req_id].append(res['data'])
    req_ids = [req_id for req_id, res in zip(req_ids, result) if not res['last']]
    input_str = [s for s, res in zip(input_str, result) if not res['last']]
    params = [p for p, res in zip(params, result) if not res['last']]
    if not req_ids:
        break

print('========== inference_infty ===========')
for req_id, out in output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{input_all[req_id][0] + ''.join(out)}\n=====")


