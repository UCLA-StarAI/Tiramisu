import torch
import pyjuice as juice
import pyjuice.nodes.distributions as dists


def construct_divisions(sid, eid, factor_half = True):
    L = eid - sid
    if L == 1:
        return [[(sid, eid)]]
    elif L == 2:
        return [[(sid, sid+1), (sid+1, eid)]]
    else:
        if factor_half:
            return [
                [(sid, sid + L // 2), (sid + L // 2, eid)],
                [(sid, sid + L // 4), (sid + L // 4, sid + L * 3 // 4), (sid + L * 3 // 4, eid)]
            ]
        else:
            return [[(sid, sid + L // 2), (sid + L // 2, eid)]]


def generate_pd_circuit(height, width, num_latents, num_cats, tie_params = True, factor_half = True, input_region_size = 1):

    range2ns = dict()
    hw2ns = dict()
    
    def recursive_construct(h_s, h_e, w_s, w_e):

        key = (h_s, h_e, w_s, w_e)
        hw = (h_e - h_s, w_e - w_s)
        if key in range2ns:
            return range2ns[key]

        if h_e - h_s == 1 and w_e - w_s == 1:
            v = h_s * width + w_s
            hwkey = ("input", h_s // input_region_size, w_s // input_region_size)
            if hwkey in hw2ns:
                target_ns = hw2ns[hwkey]
                ns = target_ns.duplicate(v, tie_params = True)
            else:
                ns = juice.inputs(
                    v, num_nodes = num_latents, 
                    dist = dists.Categorical(num_cats = num_cats)
                )
                hw2ns[hwkey] = ns
            range2ns[key] = ns
            return ns

        H = h_e - h_s
        W = w_e - w_s

        h_divisions = construct_divisions(h_s, h_e, factor_half = factor_half)
        w_divisions = construct_divisions(w_s, w_e, factor_half = factor_half)

        prod_ns = []
        for h_div in h_divisions:
            for w_div in w_divisions:

                ch_ns = []
                for h_range in h_div:
                    for w_range in w_div:
                        ns = recursive_construct(h_range[0], h_range[1], w_range[0], w_range[1])
                        ch_ns.append(ns)

                prod_ns.append(juice.multiply(*ch_ns))

        if hw in hw2ns:
            target_ns = hw2ns[hw]
            sum_ns = target_ns.duplicate(*prod_ns, tie_params = tie_params)
        else:
            edge_ids = torch.arange(0, num_latents).unsqueeze(0).repeat(2, 1)
            edge_ids[1,:] = 0
            sum_ns = juice.summate(*prod_ns, num_nodes = num_latents, edge_ids = edge_ids)
            hw2ns[hw] = sum_ns

        range2ns[key] = sum_ns

        return sum_ns

    recursive_construct(0, height, 0, width)

    root_ns = range2ns[(0, height, 0, width)]

    return root_ns
