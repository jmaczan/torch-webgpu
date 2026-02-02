"""
FX Graph transforms for fusing common patterns.

These transforms run on the FX graph BEFORE conversion to High IR,
replacing decomposed patterns with fused operations.
"""

import torch
from torch.fx import GraphModule, Node
from typing import Optional, Dict, List, Set


def find_rmsnorm_pattern(gm: GraphModule) -> List[Dict]:
    """
    Find RMSNorm patterns in the FX graph.

    RMSNorm pattern:
        variance = x.pow(2).mean(-1, keepdim=True)
        x_normed = x * rsqrt(variance + eps)
        output = x_normed * weight

    In FX graph this appears as:
        pow_node: call_method[pow](x, 2)
        mean_node: call_method[mean](pow_node, -1, keepdim=True)
        add_node: call_function[add](mean_node, eps)  # eps is a constant
        rsqrt_node: call_function[rsqrt](add_node)
        mul_norm_node: call_function[mul](x, rsqrt_node)  # or mul(rsqrt_node, x)
        mul_weight_node: call_function[mul](mul_norm_node, weight)  # or mul(weight, mul_norm_node)

    Returns list of dicts with pattern info.
    """
    patterns = []

    for node in gm.graph.nodes:
        # Start from pow(x, 2) nodes
        if not (node.op == 'call_method' and node.target == 'pow'):
            continue

        # Check pow(x, 2)
        if len(node.args) < 2 or node.args[1] != 2:
            continue

        x_input = node.args[0]
        pow_node = node

        # Find mean() that uses pow result
        mean_node = None
        for user in pow_node.users:
            if user.op == 'call_method' and user.target == 'mean':
                # Check it's mean(-1, keepdim=True)
                args = user.args
                kwargs = user.kwargs
                dim = args[1] if len(args) > 1 else kwargs.get('dim', None)
                keepdim = kwargs.get('keepdim', False) or (len(args) > 2 and args[2])
                if dim == -1 and keepdim:
                    mean_node = user
                    break

        if not mean_node:
            continue

        # Find add(mean, eps)
        add_node = None
        eps_value = None
        for user in mean_node.users:
            if user.op == 'call_function':
                target_name = getattr(user.target, '__name__', str(user.target))
                if 'add' in target_name:
                    # One arg should be mean_node, other should be eps constant
                    for arg in user.args:
                        if arg == mean_node:
                            continue
                        if isinstance(arg, (int, float)):
                            eps_value = float(arg)
                            add_node = user
                            break
                    if add_node:
                        break

        if not add_node or eps_value is None:
            continue

        # Find rsqrt(add)
        rsqrt_node = None
        for user in add_node.users:
            if user.op == 'call_function':
                target_name = getattr(user.target, '__name__', str(user.target))
                if 'rsqrt' in target_name:
                    rsqrt_node = user
                    break

        if not rsqrt_node:
            continue

        # Find mul(x, rsqrt) - the normalization multiply
        mul_norm_node = None
        for user in rsqrt_node.users:
            if user.op == 'call_function':
                target_name = getattr(user.target, '__name__', str(user.target))
                if 'mul' in target_name:
                    # Check that x_input is also an argument
                    if x_input in user.args:
                        mul_norm_node = user
                        break

        if not mul_norm_node:
            continue

        # Find mul(normalized, weight) - the weight multiply
        mul_weight_node = None
        weight_node = None
        for user in mul_norm_node.users:
            if user.op == 'call_function':
                target_name = getattr(user.target, '__name__', str(user.target))
                if 'mul' in target_name:
                    # Find the weight argument (not mul_norm_node)
                    for arg in user.args:
                        if arg != mul_norm_node:
                            weight_node = arg
                            mul_weight_node = user
                            break
                    if mul_weight_node:
                        break

        if not mul_weight_node or not weight_node:
            continue

        # Found complete pattern!
        patterns.append({
            'x_input': x_input,
            'weight': weight_node,
            'eps': eps_value,
            'pow_node': pow_node,
            'mean_node': mean_node,
            'add_node': add_node,
            'rsqrt_node': rsqrt_node,
            'mul_norm_node': mul_norm_node,
            'mul_weight_node': mul_weight_node,  # This is the output
        })

    return patterns


def fuse_rmsnorm_patterns(gm: GraphModule) -> GraphModule:
    """
    Replace RMSNorm patterns with fused torch.ops.webgpu.rms_norm calls.

    This reduces 6+ dispatches to 1 dispatch per RMSNorm.
    """
    patterns = find_rmsnorm_pattern(gm)

    if not patterns:
        return gm

    print(f"[graph_transforms] Found {len(patterns)} RMSNorm patterns to fuse")

    for pattern in patterns:
        x_input = pattern['x_input']
        weight = pattern['weight']
        eps = pattern['eps']
        output_node = pattern['mul_weight_node']

        # Insert fused op before the output node
        with gm.graph.inserting_before(output_node):
            # Create call to torch.ops.webgpu.rms_norm
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.rms_norm,
                args=(x_input, weight, eps)
            )

        # Replace all uses of the output with our fused node
        output_node.replace_all_uses_with(fused_node)

        # Remove the old nodes (in reverse order of creation)
        nodes_to_remove = [
            pattern['mul_weight_node'],
            pattern['mul_norm_node'],
            pattern['rsqrt_node'],
            pattern['add_node'],
            pattern['mean_node'],
            pattern['pow_node'],
        ]

        for node in nodes_to_remove:
            # Only remove if no remaining users
            if len(node.users) == 0:
                gm.graph.erase_node(node)

    gm.graph.lint()
    gm.recompile()

    print(f"[graph_transforms] Fused {len(patterns)} RMSNorm patterns")
    return gm


def apply_graph_transforms(gm: GraphModule) -> GraphModule:
    """Apply all graph transforms to fuse patterns."""
    gm = fuse_rmsnorm_patterns(gm)
    # Add more transforms here as we implement them
    return gm
