import itertools
import os
import pickle
import sys
import time

#
# script_directory = os.path.dirname(os.path.abspath(__file__))
# relative_path = "../../"
# new_path = os.path.normpath(os.path.join(script_directory, relative_path))
# sys.path.append(new_path)

import torch

script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../../setup"
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
    parser.add_argument('--model',
                        type=str,
                        default="gpt2")
    # Optional arguments for scanning
    parser.add_argument('--draft_model',
                        type=str,
                        default=None)

    args = parser.parse_args("")
    if varargin:
        parse_input(args, varargin)

    log_to_write = ""
    # ------------------------
    # Main
    model_id = args.model
    draft_model_id = args.draft_model

    # model, tokenizer, model_id_or_path = get_model_tokenizer(model_id, args.flash_attn.lower() == "true")

    # Test weighted request
    input_str = [
        "Hello, my name is",  # 6
        "The president of the United States is",  # 8
        "The capital of France is",  # 6
        "The future of AI is"
    ]  # 7

    batch_size = len(input_str)
    request_uids = torch.tensor(range(batch_size), device=device).view(-1, 1)


    # Parameters arguments and outputs
    spec_lengths = np.arange(10)
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

    file_name = os.path.basename(__file__)[:14]
    file_name += f"_{model_id}"
    print(directory + file_name + '.p')

    t0 = time.perf_counter()
    for idx, spec_length in enumerate(arguments):
        print_str = f"\nprocessing spec_length = {spec_length} .... \n"
        print(print_str)

        properties['spec_length'] = spec_length

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

        del runner_lmi
        import gc
        gc.collect()
        torch.cuda.empty_cache()

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
        pickle.dump([spec_lengths, output_token_latency,
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
                        # choices=['gpt2', 'bloom560', 'llama', 'llama2'],
                        default="llama")
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=1)
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=3)

    # Optional arguments for scanning
    parser.add_argument('--draft_model',
                        type=str,
                        default=None)

    # Processing
    args = parser.parse_args()

    args.model = "codellama/CodeLlama-7b-hf"
    args.draft_model = "TinyLlama/TinyLlama-1.1B-python-v0.1"

    torch.cuda.empty_cache()
    lmi_efficiency(args)
