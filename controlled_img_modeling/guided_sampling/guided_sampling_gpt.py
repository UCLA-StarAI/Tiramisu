import numpy as np
import torch
import torch.nn.functional as F
import pyjuice as juice


def guided_sampling_gpt(gpt, pc, gpt_device, pc_device, content_idx, content_idx_to_generate, condition_idx, 
                        temperature = 1.0, top_k = None, mixing_method = "likelihood_ratio", mixing_w = 1.0,
                        enable_guide = True, pc_query_interval = 5):

    idx = torch.cat((condition_idx, content_idx), dim = 1)
    idx_to_generate = torch.cat((
        torch.zeros([condition_idx.size(1)], dtype = torch.bool), content_idx_to_generate
    ), dim = 0)

    B = content_idx.size(0)
    num_cond_tokens = condition_idx.size(1)
    num_tokens = content_idx.size(1)

    p_seq = None
    p_context = None
    pc_query_cont = 0
    
    k_in, v_in = gpt.get_initial_states(idx.size(0), idx.device)
    for i in range(idx.size(1)-1):
        # forward the model to get the logits for the index in the sequence
        logits, k_in, v_in = gpt.forward_with_states(idx[:,i].unsqueeze(1), k_in, v_in)
        # pluck the logits at the final step and scale by desired temperature
        # p_{gpt} (x_i | x_{<i})
        logits = logits[:, -1, :] / temperature
        logits = F.log_softmax(logits, dim = 1)

        if enable_guide and i >= num_cond_tokens:

            cont_i = i - num_cond_tokens

            pc_idx = idx[:,num_cond_tokens:].cpu().to(pc_device)

            if p_seq is None or p_context is None or pc_query_cont >= pc_query_interval:
                # p_{pc} (x_i | x_{<i}) (note: PC is not trained on `condition_idx`)
                mask = torch.ones([num_tokens], dtype = torch.bool)
                mask[:cont_i] = False
                mask = mask[None,:].repeat(B, 1).to(pc_device)
                p_seq = juice.queries.conditional(pc, missing_mask = mask, tokens = pc_idx)
                p_seq = p_seq[:,cont_i,:].cpu().to(gpt_device)

                # p_{pc} (x_i | x_{<i}, x_{c})
                mask = torch.ones([num_tokens], dtype = torch.bool)
                mask[:cont_i] = False
                mask[~content_idx_to_generate] = False
                mask = mask[None,:].repeat(B, 1).to(pc_device)
                p_context = juice.queries.conditional(pc, missing_mask = mask, tokens = pc_idx)
                p_context = p_context[:,cont_i,:].clip(min = 1e-6).cpu().to(gpt_device)

                pc_query_interval = 1
            else:
                pc_query_interval += 1

            if mixing_method == "likelihood_ratio":
                logits = logits + mixing_w * (p_context.log() - p_seq.log())

            elif mixing_method == "geometric_mean":
                assert 0.0 < mixing_w < 1.0
                logits = (1.0 - mixing_w) * logits + mixing_w * p_context.log()

            else:
                raise ValueError()

        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # sample from the distribution
        idx_next = torch.multinomial(probs, num_samples=1)

        # append sampled index to the running sequence and continue
        if idx_to_generate[i]:
            idx[:,i+1] = idx_next.reshape(-1)

    return idx