import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from djl_python.scheduler import HuggingfaceBlock
from djl_python.scheduler.utils import compute_offsets, compute_position_ids, compute_attention_mask, merge_tensors, \
    trim_tensor, compute_kv_cache
from djl_python.scheduler.seq_batch_scheduler import SeqBatchScheduler
from djl_python.scheduler.seq_batcher_impl import ContrastiveSeqBatcher
from transformers import AutoConfig
from djl_python.scheduler.search_config import SearchConfig
import torch

from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForCausalLM


#
# tokenizer = AutoTokenizer.from_pretrained("Salesforce/xgen-7b-8k-base", trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained("Salesforce/xgen-7b-8k-base", torch_dtype=torch.bfloat16, device_map="auto")
# inputs = tokenizer("The world is", return_tensors="pt")
# sample = model.generate(**inputs, max_length=128)
# print(tokenizer.decode(sample[0]))

#%%
model_id = "mosaicml/mpt-30b"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map="auto")

lm_block = HuggingfaceBlock(model)

input0 = [
    torch.tensor([[40, 2883, 6155, 351, 616, 13779, 3290]]),
    torch.arange(7)[None, :],
    torch.ones(7, dtype=torch.int64)[None, :]
]

lm_output = lm_block.forward(*input0, None)

model_config = AutoConfig.from_pretrained(model_id)

# input with kv_cache
past_key_values = lm_output.past_key_values
input_ids = torch.tensor([[404]])
past_seq = past_key_values[0][0].shape[-2]
position_ids = torch.tensor([[past_seq]])
attention_mask = torch.ones(past_seq + 1, dtype=torch.int64)
output1 = lm_block.forward(input_ids, position_ids, attention_mask,
                           past_key_values)
