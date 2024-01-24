import itertools
import os
import pickle
import sys
import time

import seq_scheduler.utils

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)

from seq_scheduler.search_config import SearchConfig
import torch


script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../djl-serving/engines/python/setup"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)


import argparse
import numpy as np

from benchmark_utils import timeit, parse_input, PeakMemory, get_model_tokenizer, get_input_str

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def lmi_efficiency(varargin):
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=1)
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=1)
    parser.add_argument('--batch_type',
                        type=str,
                        choices=['greedy', 'contrastive'],
                        default="greedy")
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560', 'llama'],
                        default="gpt2")
    parser.add_argument("--input_size", default=2)
    parser.add_argument("--max_sparsity", default=0.001)
    parser.add_argument("--max_gen_len_size", default=2)
    parser.add_argument("--weight_choice", default=0)

    args = parser.parse_args("")
    if varargin:
        parse_input(args, varargin)

    log_to_write = ""
    # ------------------------
    # Main
    model_id = args.model
    model, tokenizer, model_id_or_path = get_model_tokenizer(model_id, args.flash_attn.lower()=="true")

    # Test weighted request
    input_str_singlets = get_input_str(args.input_size)
    weight_choice = {0: [4, 4, 3, 1, 1, 1, 0], 
                     1: [1, 1, 0, 1, 0, 0, 1],
                     2: [4, 4, 3, 1, 1, 0, 0],
                     3: [3, 2, 1, 1, 0, 0, 0]}
    weights = weight_choice[args.weight_choice][:args.input_size]
    input_str = list(itertools.chain(*[[input_str] * w for input_str, w in zip(input_str_singlets, weights)]))

    # Get input_token info
    input_ids = tokenizer(input_str, return_tensors='pt', padding=True) \
        .input_ids.view(len(input_str), -1)
    offsets = seq_scheduler.utils.compute_offsets(input_ids, [SearchConfig(pad_token_id=tokenizer.pad_token_id) for _ in range(len(input_str))])

    # Prepare input string with multiplicity
    input_str = input_str * args.concurrency
    batch_size = len(input_str)

    # Prepare requests
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    request_uids = torch.tensor(range(batch_size), device=device).view(-1, 1)

    # Parameter arguments and outputs
    max_sparsity = args.max_sparsity
    max_gen_len_array = np.array([300, 100, 50, 20])[-args.max_gen_len_size:]
    # max_gen_len_array = np.array([20, 50, 100, 300])[-args.max_gen_len_size:]
    max_splits_array = np.arange(1, args.input_size+1)

    arg_shape = len(max_gen_len_array), len(max_splits_array)

    output_token_latency = np.zeros(len(max_splits_array) * len(max_gen_len_array))
    output_seq_thruput = np.zeros(len(max_splits_array) * len(max_gen_len_array))
    output_memory = np.zeros(len(max_splits_array) * len(max_gen_len_array))

    # properties = {"tensor_parallel_degree": 1,
    #                 "dtype": "fp16",
    #                 "max_rolling_batch_size": 28,
    #                 "model_loading_timeout": 3600,
    #                 "max_rolling_batch_prefill_tokens": 1000,
    #                 "paged_attention": 'False'}

#%% patch
    properties = {
        "mpi_mode": "true",
        "tensor_parallel_degree": 1,
        "dtype": "fp16",
        "max_rolling_batch_size": 28,
        "model_loading_timeout": 3600,
        "model_id": model_id
    }
    properties["draft_model_id"] = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    properties['spec_length'] = 5

    # draft_model_id = None
    # properties["draft_model_id"] = None

    # -------- input -------
    input_str1 = [
        "Hello, my name is",  # 6
        "The president of the United States is",  # 8
        "The capital of France is",  # 6
        "The future of AI is"
    ]  # 7

    params1 = {
        "max_new_tokens": 100,
        "do_sample": False,
        "temperature": 0.001
    }

    from lmi_runner import RunnerLmi


    log_str = f"------> Namespaces: {args}, \nmax_sparsity: {max_sparsity}, max_splits_array: {max_splits_array}\n\n"
    log_str += f"max_gen_len_array: {max_gen_len_array}\n"
    log_str += f"input_token_len: {[input_ids.shape[1] - offset.item() for offset in offsets]} \n"

    print(log_str)
    log_to_write += log_str

    # Directory and file
    directory = script_directory + '/data/'
    try: os.mkdir(directory)
    except: pass

    file_name = os.path.basename(__file__)[:14]
    file_name += f"_{model_id}" + ("_flash_attn" if args.flash_attn.lower()=="true" else "") + f"_weight_choice_{args.weight_choice}_input_size_{args.input_size}"
    print(directory + file_name + '.p')

    t0 = time.perf_counter()
    for idx, max_gen_len in enumerate(max_gen_len_array):
        print_str = f"\nprocessing max_gen_len, max_gen_len = {max_gen_len} .... \n"
        print(print_str)

        # Init test kit
        param = {"max_new_tokens":max_gen_len, "do_sample":False} 
   
        runner_lmi = RunnerLmi(model_id_or_path, device, param, properties)

        @timeit(repetitions=args.reps)
        def test_run(runner, request_uids, input_str):
            return runner.pure_inference(request_uids, input_str)

        # Run the test
        avg_time, tot_gen_tokens, seq_thru_put_stat, token_latency_stat, peak_memory_stat, peak_memory2_stat, output = test_run(
            runner_lmi, request_uids, input_str)
        del runner_lmi

        print_str += \
            f"input_size: {args.input_size}" + \
            f"\navg_time: {avg_time}," + \
            f"tot_gen_tokens: {tot_gen_tokens}\n" + \
            f"seq_thru_put: {seq_thru_put_stat['avg']:.3g} reqs/sec, \n" + \
            f"token_latency: {token_latency_stat['avg']:.3g} ms/token \n" + \
            f"Peak memory usage (MiB): {peak_memory_stat['avg']}\n" + \
            f"Peak memory usage (including context) (MiB): {peak_memory2_stat['avg']}\n" + \
            "\n"

        print(print_str)
        log_to_write += print_str

        output_token_latency[idx] = token_latency_stat['avg']
        output_seq_thruput[idx] = seq_thru_put_stat['avg']
        output_memory[idx] = peak_memory_stat['avg']

    # -----------------------------------
    # Write to file
    # -----------------------------------
    output_token_latency = output_token_latency.reshape(arg_shape)
    output_seq_thruput = output_seq_thruput.reshape(arg_shape)
    output_memory = output_memory.reshape(arg_shape)
    with open(directory + file_name + '.p', 'wb') as file:
        pickle.dump([max_gen_len_array, max_splits_array, output_token_latency,
                     output_seq_thruput, output_memory], file)
        log_str = f"saved to {directory + file_name}.p\n"
        print(log_str)
        log_to_write += log_str
        
    log_str = 'Time elapse: {}s\n'.format(time.perf_counter() - t0) 
    print(log_str)
    log_to_write += log_str
    
    with open(directory + file_name + '.out', 'w') as file:
        file.write(log_to_write)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560', 'llama', 'llama2'],
                        default="llama")
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=1)
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=3)

    parser.add_argument('--max_gen_len_size', type=int, default=4)  # [300, 100, 50, 20]
    parser.add_argument("--input_size", type=int, default=6)  # [9, 44, 146, 329, 763]
                                                              # [9, 44, 146, 329, 763, 1497, 3544] 
                                                              # [9, 44, 329, 3544] 
    parser.add_argument("--weight_choice", default=0)  # [4, 4, 3, 1, 1, 1, 0], 
                                                       # [1, 1, 0, 1, 0, 0, 1]
                                                       # [4, 4, 3, 1, 1, 0, 0]
    parser.add_argument("--flash_attn", type=str, default="false") 

    args = parser.parse_args()

    os.environ["USE_FLASH_ATTENTION"] = args.flash_attn.lower()
    
    torch.cuda.empty_cache()
    lmi_efficiency(args)
