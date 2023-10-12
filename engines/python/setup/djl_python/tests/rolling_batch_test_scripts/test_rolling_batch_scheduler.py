from collections import defaultdict
import torch
from djl_python.rolling_batch.scheduler_rolling_batch import SchedulerRollingBatch
from djl_python.rolling_batch.lmi_dist_rolling_batch import LmiDistRollingBatch
import torch.distributed as dist


def print_rank0(content):
    if not dist.is_initialized() or dist.get_rank() == 0:
        print(content)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

properties = {
    "tensor_parallel_degree": 2,
    "dtype": "fp16",
    "max_rolling_batch_size": 8,
    "model_loading_timeout": 7200,
    "max_rolling_batch_prefill_tokens": 10000,
    "paged_attention": "True"
}

model_id = "openlm-research/open_llama_7b_v2"

input_str = [
    "This year's Oscar winning movie is", "This year's Oscar winning movie is"
]

params = [{
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.7,
    "seed":2023,
}, {
    "max_new_tokens": 256,
    "do_sample": False,
    "temperature": 0.7,
    "seed":20,
}]

# ===================== lmi ============================
print("=========== lmi =========")
rolling_batch = LmiDistRollingBatch(model_id, device, properties)
rolling_batch.output_formatter = None
print("reach here")

output_all = defaultdict(list)
result = rolling_batch.inference(input_str, params)
for i, res in enumerate(result):
    output_all[i].append(res['data'])

for _ in range(256):
    result = rolling_batch.inference(input_str, params)
    for i, res in enumerate(result):
        output_all[i].append(res['data'])

for i, out in enumerate(output_all.values()):
    print_rank0(input_str[i] + ''.join(out))
    print_rank0('\n====')
