import torch

import os, sys


script_directory = os.path.dirname(os.path.abspath(__file__))
relative_path = "../../../"
new_path = os.path.normpath(os.path.join(script_directory, relative_path))
sys.path.append(new_path)
sys.path.append("/usr/local/lib/python3.9/dist-packages/lmi_dist")


# Flash attention imports
from flash_attn import flash_attn_varlen_func

# lmi_vllm imports
from lmi_vllm import cache_ops
from lmi_vllm import attention_ops as lmi_vllm_attention_ops
# from lmi_vllm import cache_ops  # from vllm._C import cache_ops

# vllm imports
from vllm import attention_ops as vllm_attention_ops


def multi_query_cached_kv_attention(
    query: torch.Tensor,  # num_tokens, num_heads, head_size
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    # input_metadata
    kv_slot_mapping: torch.Tensor,
    context_lens
) -> None:
    """Multi-query attention for the generation tokens, which is essentially gather + flash-attention
    Args:
        query: shape = [num_generation_tokens, num_heads, head_size]
        key_cache: shape = [num_blocks, num_kv_heads, head_size/x,
            block_size, x]
        value_cache: shape = [num_blocks, num_kv_heads, head_size,
            block_size]
        input_metadata: metadata for paged attention.
    """
    num_query_tokens, num_heads, head_size = query.shape
    _, num_key_value_heads, _, _ = value_cache.shape
    # assert self.num_heads == num_heads
    bsz = context_lens.shape[0]
    query_sql = num_query_tokens // bsz
            
    # cumulative lengths, used to deliminate query and kv_cache
    cu_seqlens_q = torch.arange(bsz + 1, dtype=context_lens.dtype, device=context_lens.device) * query_sql
    past_kv_sql = context_lens - 1  # context_len = past_kv_sql + single_query_sql
    all_kv_sql = past_kv_sql + query_sql
    cu_seqlens_k = torch.nn.functional.pad(all_kv_sql.cumsum(0, dtype=torch.int32), (1, 0) ,value=0)   #: [16, 18, 16, 17]
    max_kv_sql = torch.max(all_kv_sql).item()

    # Allocate key and value entries
    num_kv_tokens = kv_slot_mapping.shape[0]  #: [num_tkn, ...]
    key = query.new_empty(num_kv_tokens, num_key_value_heads, head_size)
    value = query.new_empty(num_kv_tokens, num_key_value_heads, head_size)

    # Call vLLM kernel to collect kv cache entries into key and value
    cache_ops.gather_cached_kv(key, value, key_cache, value_cache, kv_slot_mapping)

    ## assert gather_cached_kv
    blk, sub_blk = kv_slot_mapping // 16, kv_slot_mapping % 16
    assert ((value - value_cache[blk, :, :, sub_blk]).abs() < 1e-6).all()

    softmax_scale = 0.125
    # softmax_scale = 0.125 - 0.001

    # Call flashattention v2
    output = flash_attn_varlen_func(
        # num_tokens, num_heads, head_size
        query.view(-1, num_heads, head_size), 
        key, 
        value,
        cu_seqlens_q, 
        cu_seqlens_k,
        max_seqlen_q=query_sql,
        max_seqlen_k=max_kv_sql,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,  # checked that flashattention uses our expected causal masking from v2.1
    )
    return output


def single_query_cached_paged_attn(
        query: torch.Tensor,
        kv_cache,
        # input_metadata
        block_tables, 
        input_lengths,  # context_lens
        max_s  # max_context_len
    ):         
        attn_output = torch.empty_like(query)
        # self attributes
        softmax_scale = 0.125
        _, num_key_value_heads, key_value_head_size, block_size = kv_cache[1].shape
        num_seqs_sqa, num_heads, head_size = query.shape
        num_key_value_groups = num_heads // num_key_value_heads
        kv_head_mapping = torch.arange(
            0, num_key_value_heads, dtype=torch.int32, device=query.device
        ).repeat_interleave(num_key_value_groups)


        # kv_cache[1]: [num_blocks, num_heads, head_size, block_size]
        block_size = kv_cache[1].shape[3]

        num_seqs_sqa, num_heads, head_size = query.shape
        max_num_partitions = ((max_s + 512 - 1) // 512)
        # use_v1 = max_num_partitions == 1 or num_seqs * num_heads > 512  # bsz used to num_seqs
        # use_v1 = use_v1 or (512 % block_size != 0)

        use_v1 = max_s <= 8192 and (
            max_num_partitions == 1 or num_seqs_sqa * num_heads > 512)

        if use_v1:
            lmi_vllm_attention_ops.single_query_cached_kv_attention(
                attn_output,
                query,
                kv_cache[0],
                kv_cache[1],
                # input_metadata
                kv_head_mapping,
                softmax_scale,
                block_tables,
                input_lengths,  # context_lens
                block_size,
                max_s  # max_context_len
            )
        else:
            # use_v2 and single query cached attn
            tmp_output = torch.empty(
                size=(num_seqs_sqa, num_heads, max_num_partitions, head_size),
                dtype=attn_output.dtype,
                device=attn_output.device,
            )
            exp_sums = torch.empty(
                size=(num_seqs_sqa, num_heads, max_num_partitions),
                dtype=torch.float32,
                device=attn_output.device,
            )
            max_logits = torch.empty_like(exp_sums)
            vllm_attention_ops.paged_attention_v2(
                attn_output,
                exp_sums,
                max_logits,
                tmp_output,
                query,
                kv_cache[0],
                kv_cache[1],
                # input_metadata
                kv_head_mapping,
                softmax_scale,
                block_tables,
                input_lengths,  # context_lens
                block_size,
                max_s,  # max_context_len
                None,  # alibi_slopes
            )

        return attn_output


def test_one_layer(file):
    load = torch.load(file)
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    query = load['query']
    kv_cache = load['kv_cache']
    block_tables = load['block_tables']
    input_lengths = load['input_lengths']  # [7, 9, 7, 8]
    max_s = input_lengths.max().item()

    # Ground truth: single-query cached paged attention
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    attn_out_sqa = single_query_cached_paged_attn(query, kv_cache, block_tables, input_lengths, max_s)

    kv_slots_mapping = torch.tensor([0,   1,   2,   3,   4,   5, 6, 
                                    224, 225, 226, 227, 228, 229, 230, 231, 232,
                                    448, 449, 450, 451, 452, 453, 454,
                                    672, 673, 674, 675, 676, 677, 678, 679], 
                                    device=query.device, dtype=torch.int32)

    # Test mq cached flash_attn
    #: [num_tkn=4, num_heads_q=32, hsz=64]
    attn_out_mqa = multi_query_cached_kv_attention(query, 
                                                   kv_cache[0],
                                                   kv_cache[1], 
                                                   kv_slots_mapping, 
                                                   input_lengths)

    assert (attn_out_sqa - attn_out_mqa).abs().max() < 5e-3

    max_diff = 0
    if not (attn_out_sqa - attn_out_mqa).abs().max() < 1e-4:
        watch = (attn_out_sqa - attn_out_mqa).abs().view(-1).topk(50)
        dbstop = 1
        diff = attn_out_sqa - attn_out_mqa
        val, idx = torch.topk(diff.abs().view(-1), 50)
        print('\n')
        print(file)
        print((attn_out_sqa - attn_out_mqa).abs().max())
        print(diff.view(-1)[idx])
        max_diff = max(max_diff, (attn_out_sqa - attn_out_mqa).abs().max())

    print('\n', max_diff)
    return max_diff

def main():
    """
    torch.save({"query": query, "kv_cache": kv_cache, "block_tables": block_tables, "input_lengths": input_lengths}, "sqa_input_0.pt")
    """
    max_err = 0
    avg_err = 0
    for i in range(22):
        file = f'./data/sqa_input_{i}.pt'
        err = test_one_layer(file)
        max_err = max(max_err, err)
        avg_err += 1/22 * err
    print(f"max_err_all_files: {max_err}")
    print(f"avg_err_all_files: {avg_err}")

"""
0.0459

0.0151
0.0146
"""
    
"""
0.0039
0.0008

0.015
0.007

0.015
0.007
"""

if __name__ == '__main__':
    main()

    # kv_slots_delta = tensor([  6, 232, 454, 679], device='cuda:0', dtype=torch.int32)
    # tensor([  0,   1,   2,   3,   4,   5, 224, 225, 226, 227, 228, 229, 230, 231,
    #     448, 449, 450, 451, 452, 453, 672, 673, 674, 675, 676, 677, 678],
    #    device='cuda:0', dtype=torch.int32)

   
    # target_kv_slots
    # tensor([105, 106, 107, 108, 109, 110, 111, 331, 332, 333, 334, 335, 336, 337,
    #         338, 339, 553, 554, 555, 556, 557, 558, 559, 778, 779, 780, 781, 782,
    #         783, 784, 785], device='cuda:0', dtype=torch.int32)

    # kv_slot_mapping = torch.zeros(num_tkn, dtype=torch.int32, device=batch.slot_indices.device)

    # target_kv_slots
    # tensor([105, 106, 107, 108, 109, 110, 111, 331, 332, 333, 334, 335, 336, 337,
    #         338, 339, 553, 554, 555, 556, 557, 558, 559, 778, 779, 780, 781, 782,
    #         783, 784, 785], device='cuda:0', dtype=torch.int32)

