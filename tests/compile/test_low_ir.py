"""Tests for Low IR op mappings."""

from torch_webgpu.compiler.high_ir import HighIROp
from torch_webgpu.compiler.low_ir import (
    LowIROp,
    high_ir_op_to_low_ir_op,
    low_ir_op_to_low_ir_node,
    LowIRNode,
)


class TestLowIROpMappings:
    """Test that all High IR ops map to Low IR ops correctly."""

    def test_all_high_ir_ops_have_low_ir_mapping(self):
        """Every High IR op should have a Low IR mapping."""
        for op in HighIROp:
            assert op in high_ir_op_to_low_ir_op, f"Missing Low IR mapping for {op}"
            low_ir_ops = high_ir_op_to_low_ir_op[op]
            assert isinstance(low_ir_ops, list), f"Low IR mapping for {op} should be a list"
            assert len(low_ir_ops) > 0, f"Low IR mapping for {op} should not be empty"

    def test_compute_ops_map_to_run_shader(self):
        """Compute operations should map to RUN_SHADER."""
        compute_ops = [
            HighIROp.ADD, HighIROp.SUB, HighIROp.MUL, HighIROp.DIV,
            HighIROp.MM, HighIROp.MATMUL,
            HighIROp.RELU, HighIROp.SILU, HighIROp.GELU,
            HighIROp.COS, HighIROp.SIN, HighIROp.EXP, HighIROp.SQRT, HighIROp.RSQRT,
            HighIROp.SOFTMAX, HighIROp.LAYER_NORM,
            HighIROp.LINEAR, HighIROp.EMBEDDING,
        ]
        for op in compute_ops:
            low_ir_ops = high_ir_op_to_low_ir_op[op]
            assert LowIROp.RUN_SHADER in low_ir_ops, f"{op} should map to RUN_SHADER"

    def test_control_flow_ops(self):
        """Control flow ops should have specific Low IR mappings."""
        assert high_ir_op_to_low_ir_op[HighIROp.MOVE_TO] == [LowIROp.MOVE_TO]
        assert high_ir_op_to_low_ir_op[HighIROp.OUTPUT] == [LowIROp.OUTPUT]

    def test_tensor_creation_ops(self):
        """Tensor creation ops should create buffers."""
        assert LowIROp.CREATE_BUFFER in high_ir_op_to_low_ir_op[HighIROp.CREATE_TENSOR]
        assert LowIROp.WRITE_BUFFER in high_ir_op_to_low_ir_op[HighIROp.CREATE_TENSOR]


class TestLowIRNodeMappings:
    """Test that all Low IR ops have corresponding node classes."""

    def test_all_low_ir_ops_have_node_classes(self):
        """Every Low IR op should have a corresponding node class."""
        for op in LowIROp:
            assert op in low_ir_op_to_low_ir_node, f"Missing node class for {op}"
            node_class = low_ir_op_to_low_ir_node[op]
            assert issubclass(node_class, LowIRNode), f"{op} node class should subclass LowIRNode"
