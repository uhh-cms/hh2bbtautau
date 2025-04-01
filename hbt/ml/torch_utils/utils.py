from __future__ import annotations
from collections import defaultdict

def reorganize_list_idx(entries):
    first = entries[0]
    if isinstance(first, int):
        return entries
    elif isinstance(first, dict):
        return reorganize_dict_idx(entries)
    elif isinstance(first, (list, tuple)):
        sub_dict = defaultdict(list)
        for e in entries:
            # only the last entry is the idx, all other entries
            # in the list/tuple will be used as keys
            sub_dict[tuple(e[:-1])].append(e[-1])
        return sub_dict

def reorganize_dict_idx(batch):
    return_dict = dict()
    for key, entries in batch.items():
        # type shouldn't change within one set of entries,
        # so just check first
        return_dict[key] = reorganize_list_idx(entries)
    return return_dict

def reorganize_idx(batch):
    if isinstance(batch, dict):
        return reorganize_dict_idx(batch)
    else:
        return reorganize_list_idx(batch)