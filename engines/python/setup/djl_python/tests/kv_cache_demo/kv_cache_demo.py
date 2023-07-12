#%% Dependencies
from djl_python.scheduler import HuggingfaceBlock
from djl_python.scheduler.utils import compute_kv_cache
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.search_config import SearchConfig
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#%% Prompt
import os
os.chdir("/Users/fenkexin/Desktop/forked/djl-serving/engines/python/setup/djl_python/tests")
file_dir = "./resources/"
file_name = "prompt3.csv"
with open(file_dir + file_name, "r") as file:
    prompt3 = file.read()


#%% Save the prompt cache into file system
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, device_map="auto")

tokenizer.pad_token = "[PAD]"
lm_block = HuggingfaceBlock(model)

# Save a kv_cache to file for later use
kv_cache_file_name = "kv_prompt.pt"
kv_cache_files = [file_dir + kv_cache_file_name]
compute_kv_cache(tokenizer.encode(prompt3, return_tensors='pt'), lm_block, kv_cache_files, search_configs=None)


#%% input string
input_string = """
Shopper: given shoppers’ short term click history
1.Chaos World Men's Novelty Hoodie Realistic 3D Print Pullover Unisex Casual Sweatshirt
2.EOWJEED Unisex Novelty 3D Printed Hoodies Long Sleeve Sweatshirts for Men Women with Big Pockets
3.McCormick All Natural Pure Vanilla Extract, 1 fl oz
4.SOLY HUX Men's Letter Graphic Hoodies Long Sleeve Drawstring Pocket Casual Pullover Sweatshirt
current search query 5xlt hoodie. 
what is the shopper’s preferred attributes the shopper is looking for given the search query? 
Assistant:
"""

#%% Load from file
config = SearchConfig(max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

kv_cache = torch.load(kv_cache_files[0])

# input
input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.to(device)
request_ids = torch.tensor([[0]])
scheduler.add_request(input_ids, request_ids, kv_cache=kv_cache)

for idx, _ in enumerate(scheduler.increment_forward(100)):
    pass
results = scheduler.results
print('------------Load from file------------------')
print(tokenizer.decode(results[0][kv_cache[0][0].shape[2]:]))

#%% runtime lru cache
scheduler = SeqBatchScheduler(lm_block, "contrastive", config)
search_configs = [SearchConfig(max_new_tokens=256,
                               eos_token_id=tokenizer.eos_token_id,
                               use_lru_kv_cache=True)]
prompt_ids = tokenizer(prompt3, return_tensors='pt', padding=True).input_ids.to(device)
prompt_ids_dict = {0: prompt_ids}

# input
input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.to(device)
request_ids = torch.tensor([[0]])

scheduler.add_request(input_ids,
                      request_ids,
                      search_configs=search_configs,
                      kv_cache_prompt_ids=prompt_ids_dict)

for idx, _ in enumerate(scheduler.increment_forward(100)):
    pass
results = scheduler.results
print('------------Runtime LRU cache------------------')
print(tokenizer.decode(results[0][kv_cache[0][0].shape[2]:]))

#%% no cache
config = SearchConfig(max_new_tokens=256, eos_token_id=tokenizer.eos_token_id)
scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

# input
prompt_ids = tokenizer(prompt3, return_tensors='pt', padding=True).input_ids.to(device)
input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.to(device)

input_ids = torch.concat([prompt_ids, input_ids], dim=1)
request_ids = torch.tensor([[0]])

scheduler.add_request(input_ids,
                      request_ids)

for idx, _ in enumerate(scheduler.increment_forward(100)):
    pass
results = scheduler.results
print('------------No cache------------------')
print(tokenizer.decode(results[0][kv_cache[0][0].shape[2]:]))

