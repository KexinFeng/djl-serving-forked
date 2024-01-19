import torch

import os, sys


script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.9/dist-packages/lmi_dist")

from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
from djl_python.tests.rolling_batch_test_scripts.generator import Generator, print_rank0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 0 if torch.cuda.is_available() else 0
"""
option.model_id=TheBloke/Llama-2-13B-Chat-fp16
option.tensor_parallel_degree=4
option.dtype=fp16
option.rolling_batch=auto
option.max_rolling_batch_size=8
option.model_loading_timeout=7200
option.max_rolling_batch_prefill_tokens=36080
"""
properties = {"mpi_mode": "true",
              "tensor_parallel_degree": 1,
              "dtype": "fp16",
              "max_rolling_batch_size": 28,
              "model_loading_timeout": 3600,
              "max_rolling_batch_prefill_tokens": 1000,
              "paged_attention": "True"}

# model_id = "TheBloke/Llama-2-13B-Chat-fp16"  # multi gpu; 7,236 MiBx4 
# model_id = "openlm-research/open_llama_7b_v2"

# model_id = "huggyllama/llama-7b"  # 9,542MiB / 23,028MiB;
# model_id = "JackFram/llama-160m"  #   844MiB / 23,028MiB;

# model_id = "bigscience/bloom-560m"  # OOM on a single gpu and not sharded on multi gpu
# model_id = "gpt2"
# model_id = "facebook/opt-125m"

model_id = "TheBloke/Llama-2-7B-Chat-fp16"  # 14,114MiB / 23,028MiB
draft_model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"  #  2,710MiB / 23,028MiB
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
# weight model.layers.0.self_attn.rotary_emb.inv_freq does not exist
# model_id = "TinyLlama/TinyLlama-1.1B-python-v0.1"
# model_id = "codellama/CodeLlama-7b-hf"  # 14,054MiB / 23028MiB;
# draft_model_id = None
properties['spec_length'] = 1

# ===================== lmi ============================
device = int(os.environ.get("RANK", 0))
# device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

properties["model_id"] = model_id
properties["draft_model_id"] = draft_model_id
properties["device"] = device
rolling_batch = LmiDistRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None

gen = Generator(rolling_batch=rolling_batch)

print('========== init inference ===========')
input_str1 = ["Hello, my name is", # 6
             "The president of the United States is", # 8
             "The capital of France is", # 6
             "The future of AI is"] # 7
params1 = [{"max_new_tokens":100, "do_sample":True, "temperature":0.001},
           {"max_new_tokens":100, "do_sample":True, "temperature":0.001},
           {"max_new_tokens":100, "do_sample":True, "temperature":0.001},
           {"max_new_tokens":100, "do_sample":True, "temperature":0.001}]
# params1 = [{"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1},
#            {"max_new_tokens":100, "do_sample":True, "temperature":1}]

gen.step(step=10, input_str_delta=input_str1, params_delta=params1)

for _ in range(1):
        print('========== inference1 ===========')
        input_str_delta = ["Hello, my name is Hello, my name is Hello, my name is Hello, my name is", # 22
                        "Hello, my name is Hello, my name is Hello, my name is"] # 17

        params_delta = [{"max_new_tokens":100, "do_sample":True, "temperature":0.001},
                        {"max_new_tokens":100, "do_sample":True, "temperature":0.001}]
        # params_delta = [{"max_new_tokens":100, "do_sample":True, "temperature":1},
        #                 {"max_new_tokens":100, "do_sample":True, "temperature":1}]

        gen.step(step=10, input_str_delta=input_str_delta, params_delta=params_delta)

"""12/07
"TheBloke/Llama-2-7B-Chat-fp16"

batch.slot_indices
tensor([ 15, 122, 227, 333, 444, 559], device='cuda:0')
batch.start_slots
tensor([  0, 105, 212, 317, 423, 543])
batch.block_tables_tensor
tensor([[ 0,  1,  2,  3,  4,  5,  6,  0],
        [ 7,  8,  9, 10, 11, 12, 13,  0],
        [14, 15, 16, 17, 18, 19, 20,  0],
        [21, 22, 23, 24, 25, 26, 27,  0],
        [28, 29, 30, 31, 32, 33, 34, 35],
        [36, 37, 38, 39, 40, 41, 42, 43]], device='cuda:0', dtype=torch.int32)
batch.input_lengths_tensor
tensor([16, 18, 16, 17, 22, 17], device='cuda:0', dtype=torch.int32)

head_slots = batch.slots[batch.slot_indices - batch.input_lengths + 1]
head_slots
tensor([  0, 112, 224, 336, 448, 576], device='cuda:0', dtype=torch.int32)
head_slots // 16
tensor([ 0,  7, 14, 21, 28, 36], device='cuda:0', dtype=torch.int32)
batch.start_slots
tensor([  0, 105, 212, 317, 423, 543])
batch.start_slots // 16
tensor([ 0,  6, 13, 19, 26, 33])

cur_slots = batch.slots[batch.slot_indices]
cur_slots
tensor([ 15, 129, 239, 352, 469, 592], device='cuda:0', dtype=torch.int32)
cur_slots //16
tensor([ 0,  8, 14, 22, 29, 37], device='cuda:0', dtype=torch.int32)
"""


"""1st
batches[0].block_tables_tensor
tensor([[ 0,  1,  2,  3,  4,  5,  6],
        [ 7,  8,  9, 10, 11, 12, 13],
        [14, 15, 16, 17, 18, 19, 20],
        [21, 22, 23, 24, 25, 26, 27]], device='cuda:0', dtype=torch.int32)
batches[1].block_tables_tensor
tensor([[ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]], device='cuda:0', dtype=torch.int32)
batches[1].slots.shape
torch.Size([235])
batches[0].slots.shape
torch.Size([423])
batches[0].slot_indices.shape
torch.Size([4])
batches[0].slot_indices
tensor([ 15, 122, 227, 333], device='cuda:0')
batches[1].slot_indices
tensor([ 21, 136], device='cuda:0')
batches[1].start_slots
tensor([  0, 120])
batches[0].start_slots
tensor([  0, 105, 212, 317])

# After concatenation
batch.block_tables_tensor.shape
torch.Size([6, 8])
batch.block_tables_tensor
tensor([[ 0,  1,  2,  3,  4,  5,  6,  0],
        [ 7,  8,  9, 10, 11, 12, 13,  0],
        [14, 15, 16, 17, 18, 19, 20,  0],
        [21, 22, 23, 24, 25, 26, 27,  0],
        [ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]], device='cuda:0', dtype=torch.int32)
batch.start_slots
tensor([  0, 105, 212, 317, 423, 543])
batch.slot_indices
tensor([ 16, 123, 228, 334, 445, 560], device='cuda:0')
batch.slots.shape
torch.Size([658])
"""


print('========== inference_infty ===========')
gen.step(step=500)
for req_id, out in gen.output_all.items():
    print_rank0(f"\n====req_id: {req_id}=====\n{gen.input_all[req_id][0] + ''.join(out)}\n")


"""
01/13/24:

After pausing at a dbstop and resuming, the bug is detected.
=======BUG DETECTED========
tensor([[ 0,  1,  2,  3,  4,  5,  6,  0],
        [ 7,  8,  9, 10, 11, 12, 13,  0],
        [14, 15, 16, 17, 18, 19, 20,  0],
        [21, 22, 23, 24, 25, 26, 27,  0],
        [ 0,  1,  2,  3,  4,  5,  6,  7],
        [ 8,  9, 10, 11, 12, 13, 14, 15]], device='cuda:0', dtype=torch.int32)

CacheManager.get_instance().free_block_mask.nonzero()[:20].view(-1)
tensor([16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33,
        34, 35])

Somehow the CacheManager.get_instance().free() inside flash_causal_lm.FlashCausalLMBatch.filter() is triggered.
"""

"""
1/17/24
batch.input_lengths_tensor = [7, 9, 7, 8]

torch.topk(out.view(4, -1), 5, dim=-1)
draft_model = None
values=tensor([[12.7734, 12.4141, 11.8750, 11.1797, 10.9141],
        [10.2969,  8.6641,  8.6562,  8.5469,  8.4688],
        [11.1484, 10.1953,  8.6250,  8.5703,  7.5781],
        [10.3828, 10.3281,  9.2031,  9.0781,  9.0156]], device='cuda:0',
       dtype=torch.float16),
indices=tensor([[10858,  7851,  1170,  8066, 17491],
        [  767,  1261,  6114, 14497,  4796],
        [29889, 29892,    13,   322,   450],
        [29892, 29889,   322, 29991,    13]], device='cuda:0'))
-------------
torch.topk(out.view(4, -1), 5, dim=-1)
target_model forward
values=tensor([[10.6016,  8.4453,  7.5977,  7.2305,  7.0352],
        [12.7734,  9.9844,  9.2891,  8.5312,  8.4453],
        [ 9.8203,  8.9141,  8.5000,  7.9102,  7.4023],
        [ 8.4844,  7.9062,  7.8906,  7.7695,  7.3203]], device='cuda:0',
       dtype=torch.float16),
indices=tensor([[ 1311, 29889, 24323, 18527, 17778],
        [ 6751,  9571, 29899,  7297, 12932],
        [   13,   450,  3681,     2, 29871],
        [  338,   376,   322,   591,   306]], device='cuda:0'))
"""

"""
The flash_attn mqa:

torch.topk(out_target.squeeze(1), 5, dim=-1)
torch.return_types.topk(
values=tensor([[10.6016,  8.4453,  7.5977,  7.2305,  7.0352],
        [12.7734,  9.9844,  9.2891,  8.5312,  8.4453],
        [ 9.8203,  8.9141,  8.5000,  7.9102,  7.4023],
        [ 8.4844,  7.9062,  7.8906,  7.7695,  7.3203]], device='cuda:0',
       dtype=torch.float16),
indices=tensor([[ 1311, 29889, 24323, 18527, 17778],
        [ 6751,  9571, 29899,  7297, 12932],
        [   13,   450,  3681,     2, 29871],
        [  338,   376,   322,   591,   306]], device='cuda:0'))
==========
The ground truth:

torch.topk(out_draft, 5, dim=-1)
torch.return_types.topk(
values=tensor([[12.7734, 12.4141, 11.8750, 11.1797, 10.9141],
        [10.2969,  8.6641,  8.6562,  8.5469,  8.4688],
        [11.1484, 10.1953,  8.6250,  8.5703,  7.5781],
        [10.3828, 10.3281,  9.2031,  9.0781,  9.0156]], device='cuda:0',
       dtype=torch.float16),
indices=tensor([[10858,  7851,  1170,  8066, 17491],
        [  767,  1261,  6114, 14497,  4796],
        [29889, 29892,    13,   322,   450],
        [29892, 29889,   322, 29991,    13]], device='cuda:0'))
"""