import pyjuice as juice
from pyjuice.nodes import CircuitNodes
from pyjuice.utils import BitSet


def pc_lvd_parser(root_ns: CircuitNodes):
    scope2group_id = dict()
    scope_groups = [] # Each scope group presents a set of scopes with tied parameters
    sgroup_partition_scopes = dict()

    def get_group_id(ns):
        if ns.is_tied():
            return scope2group_id[ns.get_source_ns().scope]
        else:
            return scope2group_id[ns.scope]

    for ns in root_ns:
        # Scope groups
        if ns.is_input() or ns.is_sum():
            if ns.is_tied():
                target_ns = ns.get_source_ns()
                if target_ns.scope not in scope2group_id:
                    scope2group_id[target_ns.scope] = len(scope_groups)
                    scope_groups.append([target_ns.scope])
                group_id = scope2group_id[target_ns.scope]
                scope_groups[group_id].append(ns.scope)
                scope2group_id[ns.scope] = group_id
            else:
                if ns.scope not in scope2group_id:
                    scope2group_id[ns.scope] = len(scope_groups)
                    scope_groups.append([ns.scope])

    for ns in root_ns:
        # Scope partition
        if ns.is_sum():
            par_group_id = get_group_id(ns)
            par_scope = ns.scope
            for prod_ns in ns.chs:
                ch_group_ids = tuple(get_group_id(ch_ns) for ch_ns in prod_ns.chs)

                key = (par_group_id, ch_group_ids)
                if key not in sgroup_partition_scopes:
                    sgroup_partition_scopes[key] = []
                
                sgroup_partition_scopes[key].append((par_scope, tuple(ch_ns.scope for ch_ns in prod_ns.chs)))

    return scope2group_id, scope_groups, sgroup_partition_scopes
