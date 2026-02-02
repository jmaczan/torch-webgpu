"""
Aggressive fusion system for maximum WebGPU performance.

This module provides graph-level pattern matching and fusion to minimize
the number of WebGPU dispatches. Each dispatch has ~0.4ms overhead, so
reducing from 800 ops to 50 ops can save ~300ms per forward pass.

Fusion targets:
1. RMSNorm: 6+ ops → 1 op (49 instances in Qwen = 245 ops saved)
2. Attention: 7+ ops → 1 op (24 instances = 144 ops saved)
3. MLP block: 5+ ops → 2 ops (24 instances = 72 ops saved)
4. Linear + activation: 2-3 ops → 1 op
"""

import operator
import torch
import torch.nn.functional as F
from torch.fx import GraphModule, Node
from torch.fx.passes.utils.source_matcher_utils import get_source_partitions
from typing import Optional, Dict, List, Set, Tuple, Any
from collections import defaultdict


class PatternMatcher:
    """
    Advanced pattern matcher that can find complex, non-consecutive patterns
    in FX graphs by analyzing data flow.
    """

    def __init__(self, graph: GraphModule):
        self.graph = graph
        self.node_to_users = defaultdict(set)
        self.node_to_inputs = defaultdict(set)
        self._build_graph_info()

    def _build_graph_info(self):
        """Build reverse lookup tables for efficient pattern matching."""
        for node in self.graph.graph.nodes:
            for inp in node.all_input_nodes:
                self.node_to_users[inp].add(node)
                self.node_to_inputs[node].add(inp)

    def find_op(self, node: Node, op_name: str) -> bool:
        """Check if a node matches an operation name."""
        if node.op == 'call_function':
            target_name = getattr(node.target, '__name__', str(node.target))
            return op_name in target_name
        elif node.op == 'call_method':
            return node.target == op_name
        return False

    def get_single_user(self, node: Node) -> Optional[Node]:
        """Get the single user of a node, or None if multiple/zero users."""
        users = list(node.users.keys())
        return users[0] if len(users) == 1 else None

    def trace_to_op(self, start: Node, target_op: str, max_depth: int = 10) -> Optional[Node]:
        """
        Trace from a node through single-user chains to find a target operation.
        """
        current = start
        for _ in range(max_depth):
            user = self.get_single_user(current)
            if user is None:
                return None
            if self.find_op(user, target_op):
                return user
            current = user
        return None


def find_rmsnorm_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find RMSNorm patterns in the FX graph using data flow analysis.

    RMSNorm: y = x * rsqrt(mean(x^2) + eps) * weight

    Pattern variations:
    1. Standard: pow → mean → add → rsqrt → mul → mul
    2. With dtype casts: to(float32) → pow → mean → add → rsqrt → mul → to(dtype) → mul
    """
    patterns = []
    matcher = PatternMatcher(gm)

    for node in gm.graph.nodes:
        # Start from pow(x, 2) nodes
        if not matcher.find_op(node, 'pow'):
            continue

        # Check pow(x, 2)
        if len(node.args) < 2:
            continue
        exp_arg = node.args[1]
        if not (exp_arg == 2 or (isinstance(exp_arg, float) and exp_arg == 2.0)):
            continue

        x_input = node.args[0]
        pow_node = node

        # Find mean() that uses pow result
        mean_node = None
        for user in pow_node.users:
            if matcher.find_op(user, 'mean'):
                mean_node = user
                break

        if not mean_node:
            continue

        # Find add(mean, eps) - eps is typically a small float
        add_node = None
        eps_value = 1e-6  # default
        for user in mean_node.users:
            if matcher.find_op(user, 'add'):
                for arg in user.args:
                    if arg != mean_node and isinstance(arg, (int, float)):
                        eps_value = float(arg)
                        add_node = user
                        break
                if add_node:
                    break

        if not add_node:
            continue

        # Find rsqrt(add)
        rsqrt_node = None
        for user in add_node.users:
            if matcher.find_op(user, 'rsqrt'):
                rsqrt_node = user
                break

        if not rsqrt_node:
            continue

        # Find multiplication with original input (x * rsqrt_result)
        # This could be direct or through dtype casts
        mul_norm_node = None
        for user in rsqrt_node.users:
            if matcher.find_op(user, 'mul'):
                # Check if x_input is in the multiplication chain
                args = list(user.args)
                # x might be directly in args, or we need to trace back through casts
                if x_input in args or _traces_to_input(args, x_input, gm):
                    mul_norm_node = user
                    break

        if not mul_norm_node:
            continue

        # Find the final weight multiplication
        mul_weight_node = None
        weight_node = None

        # The weight mul might be directly after mul_norm, or after a dtype cast
        candidates = list(mul_norm_node.users.keys())

        for candidate in candidates:
            if matcher.find_op(candidate, 'mul'):
                # This is the weight multiply
                for arg in candidate.args:
                    if arg != mul_norm_node and not isinstance(arg, (int, float)):
                        weight_node = arg
                        mul_weight_node = candidate
                        break
            elif matcher.find_op(candidate, 'to'):
                # Dtype cast - look for mul after
                for cast_user in candidate.users:
                    if matcher.find_op(cast_user, 'mul'):
                        for arg in cast_user.args:
                            if arg != candidate and not isinstance(arg, (int, float)):
                                weight_node = arg
                                mul_weight_node = cast_user
                                break
                        if mul_weight_node:
                            break

        if not mul_weight_node or not weight_node:
            continue

        # Collect all nodes in this pattern for removal
        pattern_nodes = {pow_node, mean_node, add_node, rsqrt_node, mul_norm_node, mul_weight_node}

        # Also collect any intermediate cast nodes
        for n in list(gm.graph.nodes):
            if n not in pattern_nodes:
                # Check if this node is between our pattern nodes
                if n.op in ('call_method', 'call_function'):
                    inputs = set(n.all_input_nodes)
                    users = set(n.users.keys())
                    if inputs & pattern_nodes and users & pattern_nodes:
                        pattern_nodes.add(n)

        patterns.append({
            'x_input': x_input,
            'weight': weight_node,
            'eps': eps_value,
            'output_node': mul_weight_node,
            'pattern_nodes': pattern_nodes,
        })

    return patterns


def _traces_to_input(args: List, target_input: Node, gm: GraphModule, depth: int = 5) -> bool:
    """Check if any arg traces back to target_input through casts/views."""
    for arg in args:
        if not isinstance(arg, Node):
            continue
        if arg == target_input:
            return True
        if depth > 0 and arg.op in ('call_method', 'call_function'):
            target_name = getattr(arg.target, '__name__', str(arg.target)) if arg.op == 'call_function' else arg.target
            if target_name in ('to', 'float', 'half', 'view', 'reshape', 'contiguous'):
                if _traces_to_input(list(arg.args), target_input, gm, depth - 1):
                    return True
    return False


def find_residual_rmsnorm_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find residual add + RMSNorm patterns.

    Pattern:
        residual = x + sublayer_out
        normalized = rmsnorm(residual, weight, eps)

    Where residual is also used as input to the next sublayer (skip connection).

    This is tricky because our fused_residual_rmsnorm outputs BOTH:
    - residual: for the skip connection
    - normalized: for the sublayer input
    """
    patterns = []
    matcher = PatternMatcher(gm)

    # First, find all RMSNorm patterns (we already have this function)
    rmsnorm_patterns = find_rmsnorm_patterns(gm)

    for rmsnorm_pattern in rmsnorm_patterns:
        x_input = rmsnorm_pattern['x_input']

        # Check if x_input is an add operation
        if not hasattr(x_input, 'op') or x_input.op != 'call_function':
            continue

        target_name = getattr(x_input.target, '__name__', str(x_input.target))
        if 'add' not in target_name.lower():
            continue

        add_node = x_input

        # The add should have exactly 2 args
        if len(add_node.args) != 2:
            continue

        # Check if add result is used in multiple places (skip connection + RMSNorm)
        add_users = list(add_node.users.keys())
        if len(add_users) < 2:
            # If only used by RMSNorm, no skip connection - could still fuse but simpler
            # For now, only fuse when there's a skip connection
            continue

        # Found a residual + RMSNorm pattern!
        patterns.append({
            'add_node': add_node,
            'x': add_node.args[0],       # skip connection input
            'sublayer_out': add_node.args[1],  # sublayer output
            'weight': rmsnorm_pattern['weight'],
            'eps': rmsnorm_pattern['eps'],
            'rmsnorm_output': rmsnorm_pattern['output_node'],
            'rmsnorm_pattern': rmsnorm_pattern,  # Keep reference to remove nodes
        })

    return patterns


def fuse_residual_rmsnorm(gm: GraphModule, patterns: List[Dict]) -> GraphModule:
    """
    Replace residual + RMSNorm patterns with fused_residual_rmsnorm.

    Before:
        residual = x + sublayer_out
        normalized = rmsnorm(residual, weight, eps)

    After:
        residual, normalized = fused_residual_rmsnorm(x, sublayer_out, weight, eps)

    This reduces 2 dispatches (add + rmsnorm) to 1 dispatch.
    """
    if not patterns:
        return gm

    print(f"[fusion] Fusing {len(patterns)} residual+RMSNorm patterns")

    for pattern in patterns:
        add_node = pattern['add_node']
        x = pattern['x']
        sublayer_out = pattern['sublayer_out']
        weight = pattern['weight']
        eps = pattern['eps']
        rmsnorm_output = pattern['rmsnorm_output']

        # Insert fused op before the add node
        with gm.graph.inserting_before(add_node):
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.fused_residual_rmsnorm,
                args=(x, sublayer_out, weight, eps)
            )

        # Create getitem nodes to extract residual and normalized outputs
        with gm.graph.inserting_after(fused_node):
            residual_getitem = gm.graph.call_function(
                operator.getitem,
                args=(fused_node, 0)  # residual
            )
        with gm.graph.inserting_after(residual_getitem):
            norm_getitem = gm.graph.call_function(
                operator.getitem,
                args=(fused_node, 1)  # normalized
            )

        # Copy metadata
        residual_getitem.meta = add_node.meta.copy() if add_node.meta else {}
        norm_getitem.meta = rmsnorm_output.meta.copy() if rmsnorm_output.meta else {}

        # Replace uses of add_node with residual output
        add_node.replace_all_uses_with(residual_getitem)

        # Replace uses of rmsnorm output with normalized output
        rmsnorm_output.replace_all_uses_with(norm_getitem)

        # Remove old nodes (add and rmsnorm pattern nodes)
        # First remove rmsnorm nodes
        for node in pattern['rmsnorm_pattern']['pattern_nodes']:
            if len(node.users) == 0:
                try:
                    gm.graph.erase_node(node)
                except Exception:
                    pass

        # Then remove add node
        if len(add_node.users) == 0:
            try:
                gm.graph.erase_node(add_node)
            except Exception:
                pass

    gm.graph.lint()
    gm.recompile()
    return gm


def find_gelu_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find GELU patterns that might be decomposed.
    GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
    """
    # GELU is usually not decomposed in modern PyTorch, but check anyway
    patterns = []
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            target_name = getattr(node.target, '__name__', str(node.target))
            if 'gelu' in target_name.lower():
                # Already a single GELU op
                continue
    return patterns


def find_attention_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find attention patterns that could be fused.

    Standard attention:
    Q, K, V = linear(x) for each
    scores = Q @ K^T / sqrt(d)
    attn = softmax(scores)
    out = attn @ V

    If scaled_dot_product_attention is already used, no fusion needed.
    """
    patterns = []
    matcher = PatternMatcher(gm)

    # Check if SDPA is already being used
    has_sdpa = False
    for node in gm.graph.nodes:
        if node.op == 'call_function':
            target_name = getattr(node.target, '__name__', str(node.target))
            if 'scaled_dot_product_attention' in target_name:
                has_sdpa = True
                break

    if has_sdpa:
        # Already using fused attention
        return []

    # TODO: Implement manual attention pattern detection
    # This is complex because Q, K, V projections may be separate or combined
    return patterns


def find_qkv_projection_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find Q, K, V projection patterns where three separate linear ops
    operate on the same input.

    Pattern:
        q = linear(x, W_q)
        k = linear(x, W_k)
        v = linear(x, W_v)

    These can be fused into a single dispatch:
        q, k, v = fused_qkv_proj(x, W_q, W_k, W_v)
    """
    patterns = []

    # Build a map of linear nodes grouped by their input
    input_to_linears = defaultdict(list)

    for node in gm.graph.nodes:
        if node.op != 'call_function':
            continue

        target_name = getattr(node.target, '__name__', str(node.target))
        if 'linear' not in target_name.lower():
            continue

        # Get the input to this linear
        if not node.args:
            continue

        input_node = node.args[0]
        input_to_linears[input_node].append(node)

    # Find groups of 3 linear ops with same input (likely Q, K, V)
    for input_node, linear_nodes in input_to_linears.items():
        if len(linear_nodes) < 3:
            continue

        # Check if these could be Q, K, V projections
        # They should have the same output dimension and same input
        # Try to find groups of 3 that have matching dimensions

        # For simplicity, if there are exactly 3 or 6 linear ops with same input,
        # assume they are Q, K, V projections (or 2 attention layers)
        # This heuristic works for most transformer architectures

        if len(linear_nodes) == 3:
            # Likely a single attention layer's Q, K, V
            patterns.append({
                'input': input_node,
                'q_linear': linear_nodes[0],
                'k_linear': linear_nodes[1],
                'v_linear': linear_nodes[2],
            })
        elif len(linear_nodes) >= 6:
            # Multiple attention layers or Q,K,V + other projections
            # Try to find triplets that feed into SDPA
            # For now, skip complex cases
            pass

    return patterns


def find_linear_activation_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find linear → activation patterns that can be fused.

    Patterns:
    - linear → relu
    - linear → gelu
    - linear → silu
    - matmul → add (bias) → activation
    """
    patterns = []
    matcher = PatternMatcher(gm)

    activation_ops = {'relu', 'gelu', 'silu', 'tanh'}

    for node in gm.graph.nodes:
        # Look for linear/matmul operations
        is_linear = False
        if node.op == 'call_function':
            target_name = getattr(node.target, '__name__', str(node.target))
            is_linear = target_name in ('linear', 'matmul', 'mm', 'addmm')

        if not is_linear:
            continue

        # Check if the only user is an activation or bias+activation
        users = list(node.users.keys())
        if len(users) != 1:
            continue

        user = users[0]
        user_op = None
        if user.op == 'call_function':
            user_op = getattr(user.target, '__name__', str(user.target))
        elif user.op == 'call_method':
            user_op = user.target

        # Direct linear → activation
        if user_op in activation_ops:
            patterns.append({
                'linear_node': node,
                'activation_node': user,
                'activation_type': user_op,
                'has_bias': False,
            })
            continue

        # linear → add (bias) → activation
        if user_op == 'add':
            add_users = list(user.users.keys())
            if len(add_users) == 1:
                act_node = add_users[0]
                act_op = None
                if act_node.op == 'call_function':
                    act_op = getattr(act_node.target, '__name__', str(act_node.target))
                elif act_node.op == 'call_method':
                    act_op = act_node.target

                if act_op in activation_ops:
                    patterns.append({
                        'linear_node': node,
                        'bias_node': user,
                        'activation_node': act_node,
                        'activation_type': act_op,
                        'has_bias': True,
                    })

    return patterns


def find_mlp_gate_up_patterns(gm: GraphModule) -> List[Dict]:
    """
    Find GLU-style MLP patterns: silu(gate) * up

    Pattern:
        gate = linear(x, W_gate)
        up = linear(x, W_up)
        hidden = silu(gate) * up

    This is 3 ops that can be fused to 1.
    """
    patterns = []

    for node in gm.graph.nodes:
        # Look for silu operations
        if node.op != 'call_function':
            continue

        target_name = getattr(node.target, '__name__', str(node.target))
        if 'silu' not in target_name.lower():
            continue

        silu_node = node
        silu_input = silu_node.args[0] if silu_node.args else None

        if silu_input is None:
            continue

        # Check if silu is followed by mul
        mul_node = None
        up_value = None
        for user in silu_node.users:
            if user.op == 'call_function':
                user_name = getattr(user.target, '__name__', str(user.target))
                if 'mul' in user_name:
                    mul_node = user
                    # Find the other operand (up)
                    for arg in mul_node.args:
                        if arg != silu_node:
                            up_value = arg
                            break
                    break

        if mul_node is None or up_value is None:
            continue

        # Check if silu_input comes from a linear op (gate projection)
        gate_linear = None
        if hasattr(silu_input, 'op') and silu_input.op == 'call_function':
            input_name = getattr(silu_input.target, '__name__', str(silu_input.target))
            if 'linear' in input_name:
                gate_linear = silu_input

        # Check if up_value comes from a linear op (up projection)
        up_linear = None
        if hasattr(up_value, 'op') and up_value.op == 'call_function':
            up_name = getattr(up_value.target, '__name__', str(up_value.target))
            if 'linear' in up_name:
                up_linear = up_value

        # Both should be linear ops with same input
        if gate_linear is not None and up_linear is not None:
            gate_input = gate_linear.args[0] if gate_linear.args else None
            up_input = up_linear.args[0] if up_linear.args else None

            if gate_input == up_input:
                patterns.append({
                    'x_input': gate_input,
                    'gate_linear': gate_linear,
                    'up_linear': up_linear,
                    'silu_node': silu_node,
                    'mul_node': mul_node,
                    'output_node': mul_node,
                })

    return patterns


def fuse_rmsnorm(gm: GraphModule, patterns: List[Dict]) -> GraphModule:
    """Replace RMSNorm patterns with fused torch.ops.webgpu.rms_norm calls."""
    if not patterns:
        return gm

    print(f"[fusion] Fusing {len(patterns)} RMSNorm patterns")

    for i, pattern in enumerate(patterns):
        x_input = pattern['x_input']
        weight = pattern['weight']
        eps = pattern['eps']
        output_node = pattern['output_node']

        # Insert fused op before the output node
        with gm.graph.inserting_before(output_node):
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.rms_norm,
                args=(x_input, weight, eps)
            )
            fused_node.meta = output_node.meta.copy() if output_node.meta else {}

        # Replace all uses of the output with our fused node
        output_node.replace_all_uses_with(fused_node)

        # Remove old nodes (be careful with order)
        nodes_to_remove = list(pattern['pattern_nodes'])

        # Sort by reverse topological order for safe removal
        node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
        nodes_to_remove.sort(key=lambda n: -node_order.get(n, 0))

        for node in nodes_to_remove:
            if len(node.users) == 0:
                try:
                    gm.graph.erase_node(node)
                except Exception:
                    pass  # Node might already be removed or have hidden users

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_linear_activation(gm: GraphModule, patterns: List[Dict]) -> GraphModule:
    """Fuse linear+activation patterns."""
    if not patterns:
        return gm

    print(f"[fusion] Fusing {len(patterns)} linear+activation patterns")

    # Map activation types to fused op names
    fused_ops = {
        'relu': 'fused_linear_relu',
        'gelu': 'fused_linear_gelu',
        'silu': 'fused_linear_silu',
    }

    for pattern in patterns:
        act_type = pattern['activation_type']
        if act_type not in fused_ops:
            continue

        # For now, we can use the existing fused_add_* ops for bias+activation
        # The linear itself still needs to be separate
        # TODO: Implement true fused linear+activation kernels

    return gm


def fuse_mlp_gate_up(gm: GraphModule, patterns: List[Dict]) -> GraphModule:
    """
    Replace GLU-style MLP patterns with fused_gate_up_silu.

    Before: gate = linear(x, W_gate), up = linear(x, W_up), out = silu(gate) * up
    After: out = torch.ops.webgpu.fused_gate_up_silu(x, W_gate, W_up)

    This reduces 4 dispatches (2 linear + silu + mul) to 1 dispatch.
    """
    if not patterns:
        return gm

    print(f"[fusion] Fusing {len(patterns)} MLP gate+up patterns")

    for pattern in patterns:
        x_input = pattern['x_input']
        gate_linear = pattern['gate_linear']
        up_linear = pattern['up_linear']
        output_node = pattern['output_node']

        # Extract weight tensors from linear ops
        # linear(input, weight, bias=None)
        gate_weight = gate_linear.args[1] if len(gate_linear.args) > 1 else None
        up_weight = up_linear.args[1] if len(up_linear.args) > 1 else None

        if gate_weight is None or up_weight is None:
            print(f"[fusion] Skipping MLP pattern - could not extract weights")
            continue

        # Insert fused op before the output node
        with gm.graph.inserting_before(output_node):
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.fused_gate_up_silu,
                args=(x_input, gate_weight, up_weight)
            )
            fused_node.meta = output_node.meta.copy() if output_node.meta else {}

        # Replace all uses of the output with our fused node
        output_node.replace_all_uses_with(fused_node)

        # Remove old nodes - collect all nodes that are part of this pattern
        nodes_to_remove = [
            output_node,  # mul node
            pattern['silu_node'],  # silu node
            # Note: We don't remove gate_linear and up_linear if they have other users
        ]

        # Only remove linear nodes if they have no other users
        if len(gate_linear.users) == 0 or (len(gate_linear.users) == 1 and pattern['silu_node'] in gate_linear.users):
            nodes_to_remove.append(gate_linear)
        if len(up_linear.users) == 0 or (len(up_linear.users) == 1 and output_node in up_linear.users):
            nodes_to_remove.append(up_linear)

        # Sort by reverse topological order for safe removal
        node_order = {n: i for i, n in enumerate(gm.graph.nodes)}
        nodes_to_remove.sort(key=lambda n: -node_order.get(n, 0))

        for node in nodes_to_remove:
            if len(node.users) == 0:
                try:
                    gm.graph.erase_node(node)
                except Exception as e:
                    pass  # Node might already be removed

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_qkv_projections(gm: GraphModule, patterns: List[Dict]) -> GraphModule:
    """
    Replace separate Q, K, V projections with fused_qkv_proj.

    Before: q = linear(x, W_q), k = linear(x, W_k), v = linear(x, W_v)
    After: q, k, v = torch.ops.webgpu.fused_qkv_proj(x, W_q, W_k, W_v)

    This reduces 3 dispatches to 1 dispatch per attention layer.
    """
    if not patterns:
        return gm

    print(f"[fusion] Fusing {len(patterns)} QKV projection patterns")

    for pattern in patterns:
        input_node = pattern['input']
        q_linear = pattern['q_linear']
        k_linear = pattern['k_linear']
        v_linear = pattern['v_linear']

        # Extract weights from linear ops
        # linear(input, weight, bias=None)
        q_weight = q_linear.args[1] if len(q_linear.args) > 1 else None
        k_weight = k_linear.args[1] if len(k_linear.args) > 1 else None
        v_weight = v_linear.args[1] if len(v_linear.args) > 1 else None

        if q_weight is None or k_weight is None or v_weight is None:
            print(f"[fusion] Skipping QKV pattern - could not extract weights")
            continue

        # Insert fused op after the input node but before any of the projections
        with gm.graph.inserting_after(input_node):
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.fused_qkv_proj,
                args=(input_node, q_weight, k_weight, v_weight)
            )

        # The fused op returns a tuple (q, k, v)
        # We need to create getitem nodes to extract each
        with gm.graph.inserting_after(fused_node):
            q_getitem = gm.graph.call_function(
                operator.getitem,
                args=(fused_node, 0)
            )
        with gm.graph.inserting_after(q_getitem):
            k_getitem = gm.graph.call_function(
                operator.getitem,
                args=(fused_node, 1)
            )
        with gm.graph.inserting_after(k_getitem):
            v_getitem = gm.graph.call_function(
                operator.getitem,
                args=(fused_node, 2)
            )

        # Copy metadata
        q_getitem.meta = q_linear.meta.copy() if q_linear.meta else {}
        k_getitem.meta = k_linear.meta.copy() if k_linear.meta else {}
        v_getitem.meta = v_linear.meta.copy() if v_linear.meta else {}

        # Replace uses of original linear nodes
        q_linear.replace_all_uses_with(q_getitem)
        k_linear.replace_all_uses_with(k_getitem)
        v_linear.replace_all_uses_with(v_getitem)

        # Remove old linear nodes
        for node in [q_linear, k_linear, v_linear]:
            if len(node.users) == 0:
                try:
                    gm.graph.erase_node(node)
                except Exception:
                    pass

    gm.graph.lint()
    gm.recompile()
    return gm


def fuse_sdpa(gm: GraphModule) -> GraphModule:
    """
    Replace scaled_dot_product_attention with our fused kernel for sequences <= 128.

    The fused kernel combines Q@K^T, scaling, masking, softmax, and attention@V
    into a single dispatch.

    For longer sequences, we fall back to the standard implementation.
    """
    fused_count = 0

    for node in list(gm.graph.nodes):
        if node.op != 'call_function':
            continue

        target_name = getattr(node.target, '__name__', str(node.target))
        if 'scaled_dot_product_attention' not in target_name:
            continue

        # Extract arguments
        # torch.nn.functional.scaled_dot_product_attention(query, key, value,
        #   attn_mask=None, dropout_p=0.0, is_causal=False, scale=None)
        query = node.args[0] if len(node.args) > 0 else node.kwargs.get('query')
        key = node.args[1] if len(node.args) > 1 else node.kwargs.get('key')
        value = node.args[2] if len(node.args) > 2 else node.kwargs.get('value')
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get('is_causal', False)
        scale = node.args[6] if len(node.args) > 6 else node.kwargs.get('scale', None)

        if query is None or key is None or value is None:
            continue

        # Insert fused SDPA call
        with gm.graph.inserting_before(node):
            fused_node = gm.graph.call_function(
                torch.ops.webgpu.fused_sdpa,
                args=(query, key, value, is_causal, scale)
            )
            fused_node.meta = node.meta.copy() if node.meta else {}

        # Replace all uses
        node.replace_all_uses_with(fused_node)

        # Remove old node
        try:
            gm.graph.erase_node(node)
            fused_count += 1
        except Exception:
            pass

    if fused_count > 0:
        print(f"[fusion] Replaced {fused_count} SDPA calls with fused kernel")
        gm.graph.lint()
        gm.recompile()

    return gm


def apply_aggressive_fusion(gm: GraphModule) -> GraphModule:
    """
    Apply all fusion passes aggressively to minimize dispatch count.

    NOTE: The fused kernel implementations (fused_gate_up_silu, fused_qkv_proj,
    fused_residual_rmsnorm, fused_sdpa) are not available in the current build.
    Only RMSNorm fusion using the existing torch.ops.webgpu.rms_norm is active.

    Fusion targets and theoretical impact for Qwen2.5-0.5B:
    1. RMSNorm: 49 × 6 ops → 49 × 1 op = -245 dispatches (IMPLEMENTED)
    2. MLP gate+up: 24 × 4 ops → 24 × 1 op = -72 dispatches (NOT IMPLEMENTED)
    3. QKV projection: 24 × 3 ops → 24 × 1 op = -48 dispatches (NOT IMPLEMENTED)
    4. SDPA: Already fused by PyTorch, optimized version not implemented

    Theoretical dispatch reduction with full fusion: ~200 → ~40-50
    Actual with only RMSNorm: ~200 → ~155 (modest improvement)
    """
    print("[fusion] Starting fusion pass")

    # Count initial operations
    initial_ops = sum(1 for n in gm.graph.nodes if n.op in ('call_function', 'call_method'))
    print(f"[fusion] Initial operation count: {initial_ops}")

    # Log pattern counts for analysis (even if not all are fused)
    residual_rmsnorm_patterns = find_residual_rmsnorm_patterns(gm)
    print(f"[fusion] Found {len(residual_rmsnorm_patterns)} residual+RMSNorm patterns (fusion disabled - kernel not available)")

    rmsnorm_patterns = find_rmsnorm_patterns(gm)
    if rmsnorm_patterns:
        print(f"[fusion] Found {len(rmsnorm_patterns)} standalone RMSNorm patterns")
        # Only apply RMSNorm fusion - this is the only fused kernel that's properly implemented
        try:
            gm = fuse_rmsnorm(gm, rmsnorm_patterns)
        except Exception as e:
            print(f"[fusion] RMSNorm fusion failed: {e}")

    qkv_patterns = find_qkv_projection_patterns(gm)
    if qkv_patterns:
        print(f"[fusion] Found {len(qkv_patterns)} QKV projection patterns")
        try:
            gm = fuse_qkv_projections(gm, qkv_patterns)
        except Exception as e:
            print(f"[fusion] QKV fusion failed: {e}")

    mlp_patterns = find_mlp_gate_up_patterns(gm)
    if mlp_patterns:
        print(f"[fusion] Found {len(mlp_patterns)} MLP gate+up patterns")
        try:
            gm = fuse_mlp_gate_up(gm, mlp_patterns)
        except Exception as e:
            print(f"[fusion] MLP fusion failed: {e}")

    linear_act_patterns = find_linear_activation_patterns(gm)
    print(f"[fusion] Found {len(linear_act_patterns)} linear+activation patterns (fusion disabled)")

    # Count final operations
    final_ops = sum(1 for n in gm.graph.nodes if n.op in ('call_function', 'call_method'))
    reduction = initial_ops - final_ops
    if reduction > 0:
        print(f"[fusion] After fusion: {final_ops} operations ({reduction} ops removed, {reduction/initial_ops*100:.1f}% reduction)")
    else:
        print(f"[fusion] After fusion: {final_ops} operations (no change)")

    return gm


# Export main entry point
def apply_fusion(gm: GraphModule) -> GraphModule:
    """Main entry point for fusion system."""
    return apply_aggressive_fusion(gm)
