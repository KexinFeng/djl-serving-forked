from collections import defaultdict
import torch
from djl_python.rolling_batch import LmiDistRollingBatch, SchedulerRollingBatch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from djl_python.scheduler.search_config import SearchConfig
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler



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
properties = {"tensor_parallel_degree": 1,
              "dtype": "fp16",
              "max_rolling_batch_size": 8,
              "model_loading_timeout": 7200,
              "max_rolling_batch_prefill_tokens": 10000,
              "paged_attention": "False"}


model_id = "huggyllama/llama-7b"

# Test LM_BLOCK
from djl_python.scheduler.lm_block import HuggingfaceBlock
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto" if device.type == "cuda" else "cpu")
lm_block = HuggingfaceBlock(model)

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
tokenizer.pad_token = "[PAD]"


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

input_ids = tokenizer(input_str, return_tensors='pt',
                        padding=True).input_ids.to(device)

search_config = SearchConfig(pad_token_id=tokenizer.pad_token_id)
search_config.max_new_seqlen = 50
scheduler = SeqBatchScheduler(lm_block, "greedy", search_config)                       
request_ids = torch.tensor([[0], [1], [2], [3], [4]])

scheduler.add_request(input_ids, request_ids)

for idx, _ in enumerate(scheduler.increment_forward(50)):
    pass

results = scheduler.results

print('\n', tokenizer.decode(results[0]), '\n')
print('\n', tokenizer.decode(results[1]), '\n')
print('\n', tokenizer.decode(results[2]), '\n')
print('\n', tokenizer.decode(results[3]), '\n')
print('\n', tokenizer.decode(results[4]), '\n')


"""
 <s> write a program to add two numbers in python
write a program to add two numbers in python.
Write a program to add two numbers in python.
The program should ask the user to enter two numbers and then add them.
The program should print the sum of the two numbers. 


 <s> Deep mind is a British artificial intelligence company that was founded in 2014. It is a subsidiary of Google. It is based in London, UK. It is a part of Google’s parent company Alphabet.
Deep mind is a 


 <s> Memories follow me left and right. I can’t escape them. I can’t escape the pain. I can’t escape the guilt. I can’t escape the fear. I can’t escape the anger. I can’t escape the sadness. I can’ 


 <s> When your legs don't work like they used to before And I can't sweep you off your feet anymore And I can't hold you in my arms anymore And I can't kiss you like I used to before And I can't make you feel the way I used to before And I can 


 <s> There's a time that I remember, when I did not know what I know now.
I was a young man, and I was strong.
I was a young man, and I was wrong.
I was a young man, and I was wrong.
I was a young man, and I 
"""

