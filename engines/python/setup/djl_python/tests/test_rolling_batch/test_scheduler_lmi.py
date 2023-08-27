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
              "paged_attention": "False"}

model_id = "TheBloke/Llama-2-13B-Chat-fp16"
model_id = "openlm-research/open_llama_7b_v2"
model_id = "huggyllama/llama-7b"

"""
{"inputs":"write a program to add two numbers in python","parameters":{"max_new_tokens":1000, "do_sample":true, "temperature":0.7}}
"""
input_str = [r"write a program to add two numbers in python", 
             r"Deep mind is a",
             r"Memories follow me left and right. I can",
             r"When your legs don't work like they used to before And I can't sweep you off",
             r"There's a time that I remember, when I did not know"]

params = [{"max_new_tokens":50, "do_sample":False, "temperature":0.7}, 
          {"max_new_tokens":50, "do_sample":False, "temperature":0.2},
          {"max_new_tokens":50, "do_sample":False, "temperature":0.2},
          {"max_new_tokens":50, "do_sample":False, "temperature":0.2},
          {"max_new_tokens":50, "do_sample":False, "temperature":0.2}]

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
    
for _ in range(40):
    result = rolling_batch.inference([], [])
    for i, res in enumerate(result):
        output_all[i].append(res['data']) 

for i, out in enumerate(output_all.values()):
    print_rank0(input_str[i] + ''.join(out))
    print_rank0('\n====')

"""transformers==4.30.2
write a program to add two numbers in python
2 0 1 4 - 0 1 - 2 2  ·  You  can  use  the  Python  Ar ithmetic  Oper ators  to  add ,  subtract ,  divide ,  and  multiply  numbers  in  Python .  The  Python  Ar ithmetic  Oper ators  are
xxxxxxx

Deep mind is a British  artificial  intelligence  company  that  was  acquired  by  Google  in   2 0 1 4 .  It  is  a  leader  in  the  field  of  deep  learning ,  a  form  of  machine  learning  that  is  based  on  neural  networks .  Deep  mind
xxxxxxx

Memories follow me left and right. I can’ t  escape  them .  I  can ’ t  escape  the  pain .  I  can ’ t  escape  the  gu ilt .  I  can ’ t  escape  the  shame .  I  can ’ t  escape  the  regret .  I  can ’
yyyyyyy

When your legs don't work like they used to before And I can't sweep you off your  feet  anymore  And  I  can ' t  hold  you  like  I  used  to  before  And  I  can ' t  make  you  feel  the  way  I  used  to  before  I  can ' t  make  you  feel  the  way  I  used  to
xxxxxxx

There's a time that I remember, when I did not know what  I  know  now . 
There ' s  a  place  that  I  remember ,  when  I  did  not  know  what  I  know  now . 
There ' s  a  face  that  I  remember ,  when  I  did  not  know  what
xxxxxx
"""

# transformers=4.29.2
"""
write a program to add two numbers in python
2014-01-22 · You can use the Python Arithmetic Operators to add, subtract, divide, and multiply numbers in Python. The Python Arithmetic Operators are

====
Deep mind is a British artificial intelligence company that was acquired by Google in 2014. It is a leader in the field of deep learning, a form of machine learning that is based on neural networks. Deep mind

====
Memories follow me left and right. I can’t escape them. I can’t escape the pain. I can’t escape the guilt. I can’t escape the shame. I can’t escape the regret. I can’

====
When your legs don't work like they used to before And I can't sweep you off your feet anymore And I can't hold you like I used to before And I can't make you feel the way I used to before I can't make you feel the way I used to

====
There's a time that I remember, when I did not know what I know now.
There's a place that I remember, when I did not know what I know now.
There's a face that I remember, when I did not know what
"""


