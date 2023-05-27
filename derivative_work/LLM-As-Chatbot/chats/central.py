
from chats import vicuna

def chat_stream(
    idx, local_data, user_message, state, 
    ctx_num_lconv, ctx_sum_prompt,
    res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
    sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
):
    model_type = state["model_type"]

    cs = vicuna.chat_stream(
        idx, local_data, user_message, state,
        ctx_num_lconv, ctx_sum_prompt,
        res_temp, res_topp, res_topk, res_rpen, res_mnts, res_beams, res_cache, res_sample, res_eosid, res_padid,
        sum_temp, sum_topp, sum_topk, sum_rpen, sum_mnts, sum_beams, sum_cache, sum_sample, sum_eosid, sum_padid
    )

    for idx, x in enumerate(cs):
        yield x        
        
    for idx, x in enumerate(cs):
        yield x