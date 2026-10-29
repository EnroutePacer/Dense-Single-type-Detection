"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple

try:
    from calflops import calculate_flops
except ModuleNotFoundError as exc:
    if exc.name != 'calflops':
        raise
    calculate_flops = None

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    if calculate_flops is None:
        params = sum(p.numel() for p in cfg.model.parameters())
        return params, {
            'Model FLOPs/MACs: skipped (calflops unavailable)   Params:%s' % params
        }

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()

    flops, macs, _ = calculate_flops(model=model_for_info,
                                        input_shape=input_shape,
                                        output_as_string=True,
                                        output_precision=4,
                                        print_detailed=False)
    params = sum(p.numel() for p in model_for_info.parameters())
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" %(flops, macs, params)}
