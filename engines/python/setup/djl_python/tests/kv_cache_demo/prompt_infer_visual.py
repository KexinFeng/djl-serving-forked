import argparse
import itertools
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from djl_python.scheduler import HuggingfaceBlock, BloomBlock
from djl_python.scheduler.utils import compute_kv_cache

from prompt_infer import runtime_cache_cpu, runtime_cache_cuda, file_cache, no_cache
import os
import torch

current_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_directory + "/..")
file_dir = "./resources/"

batch_sizes = [2, 10, 30]
prompt_ids = [0, 1, 2, 3]
cache_methods = [runtime_cache_cpu, runtime_cache_cuda, file_cache, no_cache]
reps = 3
model_id = "bloom-560m"
# model_id = "gpt2"

def main(args):
    N = len(prompt_ids) * len(batch_sizes) * len(cache_methods)
    result = np.zeros(N)
    result_tot = np.zeros(N)
    for idx, (prompt_id, batch_size, cache_method) in enumerate(itertools.product(prompt_ids, batch_sizes,
                                                                                  cache_methods)):
        print('---------')
        print(f"processing {idx}/{N}")
        print(f"{prompt_id}, {batch_size}, {cache_method.__name__}")
        torch.cuda.empty_cache()

        file_name = "prompt" + str(prompt_id) + ".csv"
        with open(file_dir + file_name, "r") as file:
            prompt_str = file.read()
        with open(file_dir + "inputs" + str(prompt_id) + ".csv") as file:
            input_str = file.read()

        if cache_method == file_cache:
            # compute kv cache and store it in file
            tokenizer = AutoTokenizer.from_pretrained(model_id,
                                                      trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, device_map="auto")

            tokenizer.pad_token = "[PAD]"
            lm_block = BloomBlock(model) if model_id == 'bigscience/bloom-560m' else HuggingfaceBlock(model)

            # Save a kv_cache to file for later use
            kv_cache_file_name = "kv_prompt" + str(prompt_id) + ".pt"
            kv_cache_files = [file_dir + kv_cache_file_name]
            compute_kv_cache(tokenizer.encode(prompt_str, return_tensors='pt'), lm_block, kv_cache_files,
                             search_configs=None)

            result[idx], result_tot[idx] = cache_method(kv_cache_files[0], input_str, request_vol=batch_size, reps=reps)
        else:
            result[idx], result_tot[idx] = cache_method(prompt_str, input_str, request_vol=batch_size, reps=reps)

    result = result.reshape((len(prompt_ids), len(batch_sizes), len(cache_methods)))
    result_tot = result_tot.reshape((len(prompt_ids), len(batch_sizes), len(cache_methods)))
    result = np.transpose(result, (1, 0, 2))
    result_tot = np.transpose(result_tot, (1, 0, 2))

    # Save the results
    np.save(file_dir + model_id + '_result_array.npy', result)
    np.save(file_dir + model_id + '_result_tot_array.npy', result_tot)

def visual_main(args):
    # Post process
    result = np.load('./resources2/' + model_id + '_result_array.npy')
    result_tot = np.load('./resources2/' + model_id + '_result_tot_array.npy')
    result_infer = result_tot - result

    prompt_lengths = []
    for prompt_id in prompt_ids:
        file_name = "prompt" + str(prompt_id) + ".csv"
        with open(file_dir + file_name, "r") as file:
            prompt_str = file.read()
            prompt_lengths.append(len(prompt_str))
    print(prompt_lengths)
    np.savetxt('./resources2/' + model_id  + "_result.csv", result[-1], delimiter=',')
    np.savetxt('./resources2/' + model_id  + "_result_tot.csv", result_tot[-1], delimiter=',')

    dbstop = 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prompt_inference')

    parser.add_argument('-r', '--reps', dest='reps', type=int, default=2)
    parser.add_argument('--max_gen_len', type=int, default=256)
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=2)
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560'],
                        default="bloom560")
    parser.add_argument('--batch_type',
                        type=str,
                        choices=['greedy', 'contrastive'],
                        default="greedy")
    args = parser.parse_args()

    # main(args)
    visual_main(args)

