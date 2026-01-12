"""Tests for High IR op mappings and node creation."""
import pytest
import torch
import torch.nn.functional as F
import operator

from torch_webgpu.compiler.high_ir import (
    HighIROp,
    fx_op_to_high_ir_op,
    high_ir_op_to_high_ir_node,
    HighIRNode,
)


class TestHighIROpMappings:
    """Test that all FX ops map to High IR ops correctly."""

    def test_basic_arithmetic_ops(self):
        """Test basic arithmetic FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.add] == HighIROp.ADD
        assert fx_op_to_high_ir_op[operator.add] == HighIROp.ADD
        assert fx_op_to_high_ir_op["add"] == HighIROp.ADD

        assert fx_op_to_high_ir_op[torch.sub] == HighIROp.SUB
        assert fx_op_to_high_ir_op[operator.sub] == HighIROp.SUB

        assert fx_op_to_high_ir_op[torch.mul] == HighIROp.MUL
        assert fx_op_to_high_ir_op[operator.mul] == HighIROp.MUL

        assert fx_op_to_high_ir_op[torch.div] == HighIROp.DIV
        assert fx_op_to_high_ir_op[operator.truediv] == HighIROp.DIV

        assert fx_op_to_high_ir_op[torch.neg] == HighIROp.NEG
        assert fx_op_to_high_ir_op[operator.neg] == HighIROp.NEG

    def test_matrix_ops(self):
        """Test matrix operation FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.mm] == HighIROp.MM
        assert fx_op_to_high_ir_op[torch.matmul] == HighIROp.MATMUL
        assert fx_op_to_high_ir_op[operator.matmul] == HighIROp.MATMUL

    def test_activation_ops(self):
        """Test activation function FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.relu] == HighIROp.RELU
        assert fx_op_to_high_ir_op[F.relu] == HighIROp.RELU
        assert fx_op_to_high_ir_op[F.silu] == HighIROp.SILU
        assert fx_op_to_high_ir_op[F.gelu] == HighIROp.GELU

    def test_unary_math_ops(self):
        """Test unary math FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.cos] == HighIROp.COS
        assert fx_op_to_high_ir_op[torch.sin] == HighIROp.SIN
        assert fx_op_to_high_ir_op[torch.exp] == HighIROp.EXP
        assert fx_op_to_high_ir_op[torch.sqrt] == HighIROp.SQRT
        assert fx_op_to_high_ir_op[torch.rsqrt] == HighIROp.RSQRT

    def test_reduction_ops(self):
        """Test reduction FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.sum] == HighIROp.SUM
        assert fx_op_to_high_ir_op["sum"] == HighIROp.SUM
        assert fx_op_to_high_ir_op[torch.mean] == HighIROp.MEAN
        assert fx_op_to_high_ir_op["mean"] == HighIROp.MEAN
        assert fx_op_to_high_ir_op[torch.max] == HighIROp.MAX
        assert fx_op_to_high_ir_op[torch.min] == HighIROp.MIN
        assert fx_op_to_high_ir_op[torch.argmax] == HighIROp.ARGMAX

    def test_shape_ops(self):
        """Test shape manipulation FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op["view"] == HighIROp.VIEW
        assert fx_op_to_high_ir_op["reshape"] == HighIROp.RESHAPE
        assert fx_op_to_high_ir_op["unsqueeze"] == HighIROp.UNSQUEEZE
        assert fx_op_to_high_ir_op["squeeze"] == HighIROp.SQUEEZE
        assert fx_op_to_high_ir_op["transpose"] == HighIROp.TRANSPOSE
        assert fx_op_to_high_ir_op["permute"] == HighIROp.PERMUTE
        assert fx_op_to_high_ir_op["contiguous"] == HighIROp.CONTIGUOUS
        assert fx_op_to_high_ir_op[torch.cat] == HighIROp.CAT

    def test_comparison_ops(self):
        """Test comparison FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.eq] == HighIROp.EQ
        assert fx_op_to_high_ir_op[operator.eq] == HighIROp.EQ
        assert fx_op_to_high_ir_op[torch.ne] == HighIROp.NE
        assert fx_op_to_high_ir_op[torch.lt] == HighIROp.LT
        assert fx_op_to_high_ir_op[torch.le] == HighIROp.LE
        assert fx_op_to_high_ir_op[torch.gt] == HighIROp.GT
        assert fx_op_to_high_ir_op[torch.ge] == HighIROp.GE

    def test_indexing_ops(self):
        """Test indexing FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[operator.getitem] == HighIROp.GETITEM

    def test_transformer_specific_ops(self):
        """Test transformer-specific FX ops have High IR mappings."""
        assert fx_op_to_high_ir_op[torch.softmax] == HighIROp.SOFTMAX
        assert fx_op_to_high_ir_op[F.softmax] == HighIROp.SOFTMAX
        assert fx_op_to_high_ir_op[F.layer_norm] == HighIROp.LAYER_NORM
        assert fx_op_to_high_ir_op[F.linear] == HighIROp.LINEAR
        assert fx_op_to_high_ir_op[F.embedding] == HighIROp.EMBEDDING
        assert fx_op_to_high_ir_op[F.scaled_dot_product_attention] == HighIROp.SCALED_DOT_PRODUCT_ATTENTION


class TestHighIRNodeMappings:
    """Test that all High IR ops have corresponding node classes."""

    def test_all_ops_have_node_classes(self):
        """Every High IR op should have a corresponding node class."""
        for op in HighIROp:
            assert op in high_ir_op_to_high_ir_node, f"Missing node class for {op}"
            node_class = high_ir_op_to_high_ir_node[op]
            assert issubclass(node_class, HighIRNode), f"{op} node class should subclass HighIRNode"

    def test_node_classes_have_correct_ir_op(self):
        """Each node class should have the correct ir_op attribute."""
        for op, node_class in high_ir_op_to_high_ir_node.items():
            assert hasattr(node_class, 'ir_op'), f"{node_class} missing ir_op attribute"
            assert node_class.ir_op == op, f"{node_class}.ir_op should be {op}"
