# %% Dependencies
from djl_python.scheduler import HuggingfaceBlock, BloomBlock
from djl_python.scheduler.utils import compute_kv_cache
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.search_config import SearchConfig
import torch
import time

from transformers import AutoTokenizer, AutoModelForCausalLM

#%% model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"========= device: {device} ==============")

# model_id = "bigscience/bloom-560m"
model_id = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_id,
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True).to(device)

tokenizer.pad_token = "[PAD]"
lm_block = BloomBlock(model) if model_id == 'bigscience/bloom-560m' else HuggingfaceBlock(model)

max_new_tokens = 20
# %% runtime
def runtime_cache_cuda(prompt: str, input_string: str, request_vol=10, reps=3):
    total_time = 0.0
    total_inference_time = 0.0
    for _ in range(reps):
        config = SearchConfig(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, use_lru_kv_cache=True)
        scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

        # Time it
        start_time = time.perf_counter()

        # Input
        input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.view(1, -1).to(device)
        input_ids = torch.repeat_interleave(input_ids, repeats=request_vol, dim=0)
        request_ids = torch.arange(request_vol).view(-1, 1)

        prompt_ids = tokenizer(prompt, return_tensors='pt', padding=True).input_ids.to(device)
        prompt_ids_dict = {i: prompt_ids.clone() for i in range(request_vol)}

        search_configs = [config] * request_vol
        scheduler.add_request(input_ids,
                              request_ids,
                              search_configs=search_configs,
                              kv_cache_prompt_ids=prompt_ids_dict, lru_cache_device=device)

        total_time += time.perf_counter() - start_time
        for idx, _ in enumerate(scheduler.increment_forward(500)):
            pass
        total_inference_time += time.perf_counter() - start_time

    return total_time * 1000 / reps / request_vol, total_inference_time * 1000 / reps / request_vol


# %% runtime
def runtime_cache_cpu(prompt: str, input_string: str, request_vol=10, reps=3):
    total_time = 0.0
    total_inference_time = 0.0
    for _ in range(reps):
        config = SearchConfig(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, \
                                                                           use_lru_kv_cache=True)
        scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

        # Time it
        start_time = time.perf_counter()

        # Input
        input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.view(1, -1).to(device)
        input_ids = torch.repeat_interleave(input_ids, repeats=request_vol, dim=0)
        request_ids = torch.arange(request_vol).view(-1, 1)

        prompt_ids = tokenizer(prompt, return_tensors='pt', padding=True).input_ids.to(device)
        prompt_ids_dict = {i: prompt_ids.clone() for i in range(request_vol)}

        search_configs = [config] * request_vol
        scheduler.add_request(input_ids,
                              request_ids,
                              search_configs=search_configs,
                              kv_cache_prompt_ids=prompt_ids_dict)

        total_time += time.perf_counter() - start_time
        for idx, _ in enumerate(scheduler.increment_forward(500)):
            pass
        total_inference_time += time.perf_counter() - start_time

    return total_time * 1000 / reps / request_vol, total_inference_time * 1000 / reps / request_vol


# %%
def file_cache(kv_cache_file: str, input_string: str, request_vol=10, reps=3):
    total_time = 0.0
    total_inference_time = 0.0
    for _ in range(reps):
        config = SearchConfig(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, use_lru_kv_cache=True)
        scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

        # Time it
        start_time = time.perf_counter()

        # input
        input_ids = tokenizer(input_string, return_tensors='pt', padding=True).input_ids.view(1, -1).to(device)
        input_ids = torch.repeat_interleave(input_ids, repeats=request_vol, dim=0)
        request_ids = torch.arange(request_vol).view(-1, 1)

        kv_cache = torch.load(kv_cache_file)
        scheduler.add_request(input_ids, request_ids, kv_cache=kv_cache)

        total_time += time.perf_counter() - start_time
        for idx, _ in enumerate(scheduler.increment_forward(500)):
            pass
        total_inference_time += time.perf_counter() - start_time

    return total_time * 1000 / reps / request_vol, total_inference_time * 1000 / reps / request_vol


# %%
def no_cache(prompt: str, input_string: str, request_vol=10, reps=3):
    total_time = 0.0
    total_inference_time = 0.0
    for _ in range(reps):
        config = SearchConfig(max_new_tokens=max_new_tokens, eos_token_id=tokenizer.eos_token_id, use_lru_kv_cache=True)
        scheduler = SeqBatchScheduler(lm_block, "contrastive", config)

        # Time it
        start_time = time.perf_counter()

        # input
        input_ids = tokenizer(prompt + input_string, return_tensors='pt', padding=True).input_ids.view(1, -1).to(device)
        input_ids = torch.repeat_interleave(input_ids, repeats=request_vol, dim=0)
        request_ids = torch.arange(request_vol).view(-1, 1)

        scheduler.add_request(input_ids, request_ids)

        total_time += time.perf_counter() - start_time
        for idx, _ in enumerate(scheduler.increment_forward(500)):
            pass
        total_inference_time += time.perf_counter() - start_time

    return total_time * 1000 / reps / request_vol, total_inference_time * 1000 / reps / request_vol


if __name__ == '__main__':
    import os

    os.chdir("/Users/fenkexin/Desktop/forked/djl-serving/engines/python/setup/djl_python/tests")
    file_dir = "./resources/"
    file_name = "prompt4.csv"
    with open(file_dir + file_name, "r") as file:
        prompt4 = file.read()

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

    file_dir = "./resources/"
    kv_cache_file_name = "kv_prompt.pt"
    kv_cache_files = [file_dir + kv_cache_file_name]

    request_vol = 10
    reps = 3

    print(f"Runtime cache, per request time is {runtime_cache_cuda(prompt4, input_string, request_vol, reps)}")
    print(f"File system cache, per request time is {file_cache(kv_cache_files[0], input_string, request_vol, reps)}")
    print(f"No cache, per request time is {no_cache(prompt4, input_string, request_vol, reps)}")
