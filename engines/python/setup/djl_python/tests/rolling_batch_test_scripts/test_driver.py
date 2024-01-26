import argparse
import torch
from lmi_perform_benchmark import lmi_efficiency

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

    args = parser.parse_args("")
    
    ## llama
    args.draft_model = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
    models = [
            #   "TheBloke/Llama-2-70B-Chat-fp16", 
              "TheBloke/Llama-2-13B-Chat-fp16", 
              "TheBloke/Llama-2-7B-Chat-fp16"
              ]
    bss = [1, 4, 8, 16, 32, 64, 128][::-1]
    for model in models:
        for bs in bss:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                print('\n')
                print(f"model= {model}, bs = {bs}, '----->'")
                args.model = model
                args.concurrency = bs

                lmi_efficiency(args)

