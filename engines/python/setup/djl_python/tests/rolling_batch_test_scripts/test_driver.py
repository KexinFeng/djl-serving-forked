import argparse
import torch
from lmi_perform_benchmark import lmi_efficiency

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Benchmark')
    parser.add_argument('--model',
                        type=str,
                        choices=['gpt2', 'bloom560', 'llama', 'llama2'],
                        default="llama2")
    parser.add_argument('-c',
                        '--concurrency',
                        dest='concurrency',
                        type=int,
                        default=1)
    parser.add_argument('-r', '--reps', dest='reps', type=int, default=3)

    parser.add_argument('--max_gen_len_size', type=int, default=4)  # [300, 100, 50, 20]
    parser.add_argument("--input_size", type=int, default=5)  # [8, 40, 127, 285, 635]
                                                              # [9, 44, 146, 329, 763, 2054]
                                                              # [9, 44, 146, 329, 763, 1777] 
    parser.add_argument("--flash_attn", type=str, default="true") 

    args = parser.parse_args("")
    
    ## llama
    flash_attns = ['true', 'false']
    models = ['llama2', 'llama']
    input_data = [(0, 6), (1, 7)]
    for model in models:
        for flash_attn in flash_attns:
            if 'llama' not in model and flash_attn == 'true': continue
            for weight_choice, input_size in input_data:
                torch.cuda.empty_cache()
                print('\n')
                print(model, flash_attn, weight_choice)
                args.flash_attn = flash_attn
                args.model = model
                args.weight_choice = weight_choice
                args.input_size = input_size
                    
                # lmi_dist.models.causal_lm.CausalLM
                # lmi_dist.models.flash_llama.FlashLlama
                args.max_gen_len_size = 3
                lmi_efficiency(args)

                # transformers.LlamaForCausalLM
                # args.max_gen_len_size = 4
                # efficiency(args)
