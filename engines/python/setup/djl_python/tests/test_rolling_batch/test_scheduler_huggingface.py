from collections import defaultdict
import torch

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

# params = [{"max_new_tokens":50, "do_sample":False, "temperature":0.7}, 
#           {"max_new_tokens":60, "do_sample":True, "temperature":0.2, "top_p": 0.9}]

# ====================== huggingface ====================
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto" if device.type == "cuda" else "cpu")
tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side='left')
tokenizer.pad_token = "[PAD]"

input_ids = tokenizer(input_str, return_tensors='pt',
                        padding=True).input_ids.to(device)

greedy_output = model.generate(input_ids, max_length=50)

print("\nOutput huggingface:\n" + 100 * '-')
for i in range(len(input_str)):
    print(tokenizer.decode(greedy_output[i], skip_special_tokens=False))
    print('\n')


"""
<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> write a program to add two numbers in python
write a program to add two numbers in python.
Write a program to add two numbers in python.
The program should ask the


<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> Deep mind is a British artificial intelligence company that was founded in 2014. It is a subsidiary of Google. It is based in London


<unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><unk><s> Memories follow me left and right. I can’t escape them. I can’t escape the pain. I can‘t escape the guilt. I can’t escape the




<s> When your legs don't work like they used to before And I can't sweep you off your feet anymore And I can't hold you in my arms anymore And I can't kiss you like I used to before And I


<unk><unk><unk><unk><unk><unk><unk><s> There's a time that I remember, when I did not know what I know now.
I was a young man, and I was strong.
I was a young man, and I was wrong
"""

