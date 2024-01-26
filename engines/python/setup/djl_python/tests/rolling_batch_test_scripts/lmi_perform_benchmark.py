import re
import os
import pickle
import sys
import time

import torch
import torch.distributed as dist

from djl_python.tests.rolling_batch_test_scripts.generator import print_rank0

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../../setup"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.10/dist-packages/lmi_dist")

import argparse
import numpy as np

from benchmark_utils import timeit, parse_input

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def lmi_efficiency(varargin):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',
                        type=str,
                        default="llama")
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        choices=[1, 4, 32, 64],
                        default=1)
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=3)

    # Optional arguments for scanning
    parser.add_argument('--draft_model',
                        type=str,
                        default=None)
    
    parser.add_argument('-s', '--size', dest='size', type=int, default=5)

    args = parser.parse_args("")
    if varargin:
        parse_input(args, varargin)

    log_to_write = ""
    # ------------------------
    # Main
    model_id = args.model
    draft_model_id = args.draft_model

    # Test weighted request
    input_str = ([
        "Write a program to add two numbers in python",
        "Write a program to add two numbers in c++",
        "Hello, my name is",  # 6
        "The president of the United States is",  # 8
        "The capital of France is",  # 6
        "The future of AI is"  # 7
    ]*10)[:args.concurrency]

    batch_size = len(input_str)
    request_uids = torch.tensor(range(batch_size), device=device).view(-1, 1)


    # Parameters arguments and outputs
    spec_lengths = np.arange(args.size)
    arguments = spec_lengths

    arg_shape = len(arguments)

    output_token_latency = np.zeros(len(arguments))
    output_seq_thruput = np.zeros(len(arguments))
    output_memory = np.zeros(len(arguments))
    # %% patch
    properties = {"mpi_mode": "true",
                  "tensor_parallel_degree": 1,
                  "dtype": "fp16", "max_rolling_batch_size": 28,
                  "model_loading_timeout": 3600}

    # draft_model_id = None
    properties["draft_model_id"] = draft_model_id
    properties['model_id'] = model_id

    # ------- Runner --------
    from lmi_runner import RunnerLmi

    log_str = ""
    print(log_str)
    log_to_write += log_str

    # Directory and file
    directory = script_directory + '/data/'
    try:
        os.mkdir(directory)
    except:
        pass

    file_name = "_".join(re.split(r'[_.]', os.path.basename(__file__))[:2])
    file_name += f"+{model_id.split('/')[1]}++{draft_model_id.split('/')[1]}"
    print(directory + file_name + '.p')

    t0 = time.perf_counter()
    for idx, spec_length in enumerate(arguments):
        print_str = f"\nprocessing spec_length = {spec_length} .... \n"
        print_rank0(print_str)

        properties['spec_length'] = spec_length
        if spec_length == 0:
            properties['draft_model_id'] = None
        else:
            properties['draft_model_id'] = draft_model_id

        # Init test kit
        param = {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": 0.001
        }

        runner_lmi = RunnerLmi(model_id, device, param, properties)

        @timeit(repetitions=args.reps)
        def test_run(runner, request_uids, input_str):
            return runner.pure_inference(request_uids, input_str)

        # Run the test
        avg_time, tot_gen_tokens, seq_thru_put_stat, token_latency_stat, peak_memory_stat, peak_memory2_stat, output = test_run(
            runner_lmi, request_uids, input_str)
        
        runner_lmi.release_cache()
        del runner_lmi
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        print_str += \
            f"tot_gen_tokens: {tot_gen_tokens}\n" + \
            f"seq_thru_put: {seq_thru_put_stat['avg']:.3g} reqs/sec, \n" + \
            f"token_latency: {token_latency_stat['avg']:.3g} ms/token \n" + \
            f"Peak memory usage (MiB): {peak_memory_stat['avg']}\n" + \
            f"Peak memory usage (including context) (MiB): {peak_memory2_stat['avg']}\n" + \
            f"input_size: {args.size}" + f"\navg_time: {avg_time}," + \
            "\n"

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(print_str)
        log_to_write += print_str

        output_token_latency[idx] = token_latency_stat['avg']
        output_seq_thruput[idx] = seq_thru_put_stat['avg']
        output_memory[idx] = peak_memory_stat['avg']

    # -----------------------------------
    # Write to file
    # -----------------------------------
    if not dist.is_initialized() or dist.get_rank() == 0:
        output_token_latency = output_token_latency.reshape(arg_shape)
        output_seq_thruput = output_seq_thruput.reshape(arg_shape)
        output_memory = output_memory.reshape(arg_shape)
        with open(directory + file_name + '.p', 'wb') as file:
            pickle.dump([arguments, output_token_latency,
                        output_seq_thruput, output_memory, 
                        model_id, draft_model_id, input_str], file)
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
                        default="llama")
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        choices=[1, 4, 32, 64],
                        default=1)
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=3)

    # Optional arguments for scanning
    parser.add_argument('--draft_model',
                        type=str,
                        default=None)
    
    parser.add_argument('-s', '--size', dest='size', type=int, default=5)
    
    # Processing
    args = parser.parse_args()

    args.model = "codellama/CodeLlama-7b-hf"
    args.model = "TheBloke/Llama-2-70B-Chat-fp16"
    args.model = "TheBloke/Llama-2-13B-Chat-fp16"
    args.draft_model = "TinyLlama/TinyLlama-1.1B-python-v0.1"

    torch.cuda.empty_cache()
    lmi_efficiency(args)
