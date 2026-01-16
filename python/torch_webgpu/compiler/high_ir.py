from typing import Any, List, Optional
import torch
from enum import StrEnum, auto

from .compiler_pass import CompilerPass, Transform, Pattern
from .ir import IRNode
from .logger import debug


class HighIROp(StrEnum):
    CREATE_TENSOR = auto()
    PLACEHOLDER = auto()  # Input tensors and model parameters
    GETATTR = auto()  # Attribute access
    ADD = auto()
    RELU = auto()
    FUSED_ADD_RELU = auto()
    FUSED_MUL_SILU = auto()
    FUSED_ADD_SILU = auto()
    FUSED_ADD_GELU = auto()
    MOVE_TO = auto()
    OUTPUT = auto()
    MUL = auto()
    MM = auto()
    # Transformer ops
    SUB = auto()
    DIV = auto()
    MATMUL = auto()
    TRANSPOSE = auto()
    CAT = auto()
    COS = auto()
    SIN = auto()
    RSQRT = auto()
    POW = auto()
    MEAN = auto()
    SOFTMAX = auto()
    SILU = auto()
    LINEAR = auto()
    EMBEDDING = auto()
    LAYER_NORM = auto()
    GETITEM = auto()
    NEG = auto()
    EXP = auto()
    SUM = auto()
    SQRT = auto()
    TANH = auto()
    GELU = auto()
    VIEW = auto()
    RESHAPE = auto()
    UNSQUEEZE = auto()
    SQUEEZE = auto()
    PERMUTE = auto()
    CONTIGUOUS = auto()
    CLONE = auto()
    EXPAND = auto()
    ARANGE = auto()
    FULL = auto()
    ZEROS = auto()
    ONES = auto()
    WHERE = auto()
    MAX = auto()
    MIN = auto()
    ARGMAX = auto()
    EQ = auto()
    NE = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    MASKED_FILL = auto()
    TRIU = auto()
    DROPOUT = auto()
    SCALED_DOT_PRODUCT_ATTENTION = auto()
    SLICE = auto()
    SELECT = auto()
    INDEX = auto()
    CUMSUM = auto()
    CAST = auto()  # dtype casting with .to(dtype)
    REPEAT_INTERLEAVE = auto()  # For GQA attention
    SET_GRAD_ENABLED = auto()  # torch.no_grad() context
    ITEM = auto()  # tensor.item() - extract scalar value
    # MoE (Mixture of Experts) ops
    TOPK = auto()  # top-k selection for expert routing
    SCATTER = auto()  # scatter values to indices
    SCATTER_ADD = auto()  # scatter with addition
    GATHER = auto()  # gather values from indices
    ANY = auto()  # any reduction (for masks)
    # vmap batch operations
    ADD_BATCH_DIM = auto()
    REMOVE_BATCH_DIM = auto()
    VMAP_INCREMENT_NESTING = auto()
    VMAP_DECREMENT_NESTING = auto()
    # Other internal ops
    LAZY_LOAD_DECOMPOSITIONS = auto()
    ENTER_AUTOCAST = auto()
    EXIT_AUTOCAST = auto()
    LOG_API_USAGE = auto()


class HighIRNode(IRNode):
    ir_op: HighIROp
    value_id: Any = None
    inputs: List[Any] = []

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        self.fx_node = fx_node
        if value_id:
            self.value_id = value_id
        if inputs:
            self.inputs = inputs

    def __str__(self):
        return self.ir_op


class HighIRCreateTensor(HighIRNode):
    ir_op = HighIROp.CREATE_TENSOR
    shape = None
    stride = None
    constant_data = None
    data = None
    dtype = None
    device = None
    numel = None
    size = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)

        self.shape = fx_node.meta["example_value"].shape
        self.dtype = fx_node.meta["example_value"].dtype
        self.device = fx_node.meta["example_value"].device
        self.numel = fx_node.meta["example_value"].itemsize
        self.stride = fx_node.meta["example_value"].stride()
        self.data = fx_node.meta["example_value"].data
        # TODO: put better checks here, because it's possible that
        # fx_node.args[0] is not a constant data I expect it to be
        if len(fx_node.args) == 1 and isinstance(fx_node.args[0], list):
            self.constant_data = fx_node.args[0]
        self.size = fx_node.meta["example_value"].size()


class HighIRPlaceholder(HighIRNode):
    """Represents input tensors and model parameters passed to the graph."""
    ir_op = HighIROp.PLACEHOLDER
    shape = None
    dtype = None
    device = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)
        if "example_value" in fx_node.meta:
            example = fx_node.meta["example_value"]
            if hasattr(example, 'shape'):
                self.shape = example.shape
                self.dtype = example.dtype
                self.device = example.device


class HighIRGetattr(HighIRNode):
    """Represents attribute access on a module."""
    ir_op = HighIROp.GETATTR


class HighIRAdd(HighIRNode):
    ir_op = HighIROp.ADD


class HighIRRelu(HighIRNode):
    ir_op = HighIROp.RELU


class HighIRFusedAddRelu(HighIRNode):
    ir_op = HighIROp.FUSED_ADD_RELU


class HighIRFusedMulSilu(HighIRNode):
    ir_op = HighIROp.FUSED_MUL_SILU


class HighIRFusedAddSilu(HighIRNode):
    ir_op = HighIROp.FUSED_ADD_SILU


class HighIRFusedAddGelu(HighIRNode):
    ir_op = HighIROp.FUSED_ADD_GELU


class HighIRMoveTo(HighIRNode):
    ir_op = HighIROp.MOVE_TO


class HighIROutput(HighIRNode):
    ir_op = HighIROp.OUTPUT


class HighIRMul(HighIRNode):
    ir_op = HighIROp.MUL


class HighIRMM(HighIRNode):
    ir_op = HighIROp.MM


class HighIRSub(HighIRNode):
    ir_op = HighIROp.SUB


class HighIRDiv(HighIRNode):
    ir_op = HighIROp.DIV


class HighIRMatmul(HighIRNode):
    ir_op = HighIROp.MATMUL


class HighIRTranspose(HighIRNode):
    ir_op = HighIROp.TRANSPOSE


class HighIRCat(HighIRNode):
    ir_op = HighIROp.CAT


class HighIRCos(HighIRNode):
    ir_op = HighIROp.COS


class HighIRSin(HighIRNode):
    ir_op = HighIROp.SIN


class HighIRRsqrt(HighIRNode):
    ir_op = HighIROp.RSQRT


class HighIRPow(HighIRNode):
    ir_op = HighIROp.POW


class HighIRMean(HighIRNode):
    ir_op = HighIROp.MEAN


class HighIRSoftmax(HighIRNode):
    ir_op = HighIROp.SOFTMAX


class HighIRSilu(HighIRNode):
    ir_op = HighIROp.SILU


class HighIRLinear(HighIRNode):
    ir_op = HighIROp.LINEAR


class HighIREmbedding(HighIRNode):
    ir_op = HighIROp.EMBEDDING


class HighIRLayerNorm(HighIRNode):
    ir_op = HighIROp.LAYER_NORM


class HighIRGetitem(HighIRNode):
    ir_op = HighIROp.GETITEM


class HighIRNeg(HighIRNode):
    ir_op = HighIROp.NEG


class HighIRExp(HighIRNode):
    ir_op = HighIROp.EXP


class HighIRSum(HighIRNode):
    ir_op = HighIROp.SUM


class HighIRSqrt(HighIRNode):
    ir_op = HighIROp.SQRT


class HighIRTanh(HighIRNode):
    ir_op = HighIROp.TANH


class HighIRGelu(HighIRNode):
    ir_op = HighIROp.GELU


class HighIRView(HighIRNode):
    ir_op = HighIROp.VIEW


class HighIRReshape(HighIRNode):
    ir_op = HighIROp.RESHAPE


class HighIRUnsqueeze(HighIRNode):
    ir_op = HighIROp.UNSQUEEZE


class HighIRSqueeze(HighIRNode):
    ir_op = HighIROp.SQUEEZE


class HighIRPermute(HighIRNode):
    ir_op = HighIROp.PERMUTE


class HighIRContiguous(HighIRNode):
    ir_op = HighIROp.CONTIGUOUS


class HighIRClone(HighIRNode):
    ir_op = HighIROp.CLONE


class HighIRExpand(HighIRNode):
    ir_op = HighIROp.EXPAND


class HighIRArange(HighIRNode):
    ir_op = HighIROp.ARANGE
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)
        if "example_value" in fx_node.meta:
            example = fx_node.meta["example_value"]
            self.shape = example.shape
            self.dtype = example.dtype
            self.device = example.device
            self.numel = example.numel()
            self.stride = example.stride()
            self.size = example.size()


class HighIRFull(HighIRNode):
    ir_op = HighIROp.FULL
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)
        if "example_value" in fx_node.meta:
            example = fx_node.meta["example_value"]
            self.shape = example.shape
            self.dtype = example.dtype
            self.device = example.device
            self.numel = example.numel()
            self.stride = example.stride()
            self.size = example.size()


class HighIRZeros(HighIRNode):
    ir_op = HighIROp.ZEROS
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)
        if "example_value" in fx_node.meta:
            example = fx_node.meta["example_value"]
            self.shape = example.shape
            self.dtype = example.dtype
            self.device = example.device
            self.numel = example.numel()
            self.stride = example.stride()
            self.size = example.size()


class HighIROnes(HighIRNode):
    ir_op = HighIROp.ONES
    shape = None
    stride = None
    dtype = None
    device = None
    numel = None
    size = None

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
    ):
        super().__init__(fx_node=fx_node, value_id=value_id, inputs=inputs)
        if "example_value" in fx_node.meta:
            example = fx_node.meta["example_value"]
            self.shape = example.shape
            self.dtype = example.dtype
            self.device = example.device
            self.numel = example.numel()
            self.stride = example.stride()
            self.size = example.size()


class HighIRWhere(HighIRNode):
    ir_op = HighIROp.WHERE


class HighIRMax(HighIRNode):
    ir_op = HighIROp.MAX


class HighIRMin(HighIRNode):
    ir_op = HighIROp.MIN


class HighIRArgmax(HighIRNode):
    ir_op = HighIROp.ARGMAX


class HighIREq(HighIRNode):
    ir_op = HighIROp.EQ


class HighIRNe(HighIRNode):
    ir_op = HighIROp.NE


class HighIRLt(HighIRNode):
    ir_op = HighIROp.LT


class HighIRLe(HighIRNode):
    ir_op = HighIROp.LE


class HighIRGt(HighIRNode):
    ir_op = HighIROp.GT


class HighIRGe(HighIRNode):
    ir_op = HighIROp.GE


class HighIRMaskedFill(HighIRNode):
    ir_op = HighIROp.MASKED_FILL


class HighIRTriu(HighIRNode):
    ir_op = HighIROp.TRIU


class HighIRDropout(HighIRNode):
    ir_op = HighIROp.DROPOUT


class HighIRScaledDotProductAttention(HighIRNode):
    ir_op = HighIROp.SCALED_DOT_PRODUCT_ATTENTION


class HighIRSlice(HighIRNode):
    ir_op = HighIROp.SLICE


class HighIRSelect(HighIRNode):
    ir_op = HighIROp.SELECT


class HighIRIndex(HighIRNode):
    ir_op = HighIROp.INDEX


class HighIRCumsum(HighIRNode):
    ir_op = HighIROp.CUMSUM


class HighIRRepeatInterleave(HighIRNode):
    ir_op = HighIROp.REPEAT_INTERLEAVE


class HighIRSetGradEnabled(HighIRNode):
    ir_op = HighIROp.SET_GRAD_ENABLED


class HighIRCast(HighIRNode):
    ir_op = HighIROp.CAST
    cast_method = None  # Original cast method: "float", "half", "int", "long", "bool", or "to"

    def __init__(
        self,
        fx_node: torch.fx.Node,
        value_id: Any = None,
        inputs: List[Any] = [],
        cast_method: str = None,
    ):
        super().__init__(fx_node, value_id, inputs)
        self.cast_method = cast_method


class HighIRItem(HighIRNode):
    """tensor.item() - extract a single scalar value from a tensor."""
    ir_op = HighIROp.ITEM


class HighIRTopk(HighIRNode):
    """torch.topk - returns top k values and indices."""
    ir_op = HighIROp.TOPK


class HighIRScatter(HighIRNode):
    """torch.scatter - scatter values to indices."""
    ir_op = HighIROp.SCATTER


class HighIRScatterAdd(HighIRNode):
    """torch.scatter_add - scatter with addition."""
    ir_op = HighIROp.SCATTER_ADD


class HighIRGather(HighIRNode):
    """torch.gather - gather values from indices."""
    ir_op = HighIROp.GATHER


class HighIRAny(HighIRNode):
    """torch.any - any reduction."""
    ir_op = HighIROp.ANY


class HighIRAddBatchDim(HighIRNode):
    ir_op = HighIROp.ADD_BATCH_DIM


class HighIRRemoveBatchDim(HighIRNode):
    ir_op = HighIROp.REMOVE_BATCH_DIM


class HighIRVmapIncrementNesting(HighIRNode):
    ir_op = HighIROp.VMAP_INCREMENT_NESTING


class HighIRVmapDecrementNesting(HighIRNode):
    ir_op = HighIROp.VMAP_DECREMENT_NESTING


class HighIRLazyLoadDecompositions(HighIRNode):
    ir_op = HighIROp.LAZY_LOAD_DECOMPOSITIONS


class HighIREnterAutocast(HighIRNode):
    ir_op = HighIROp.ENTER_AUTOCAST


class HighIRExitAutocast(HighIRNode):
    ir_op = HighIROp.EXIT_AUTOCAST


class HighIRLogApiUsage(HighIRNode):
    ir_op = HighIROp.LOG_API_USAGE


import operator
import torch.nn.functional as F
from torch._functorch.predispatch import (
    lazy_load_decompositions,
    _vmap_increment_nesting,
    _vmap_decrement_nesting,
    _add_batch_dim,
    _remove_batch_dim,
)
from torch.amp.autocast_mode import _enter_autocast, _exit_autocast

fx_op_to_high_ir_op: dict[Any, HighIROp] = {
    # Tensor creation
    torch.tensor: HighIROp.CREATE_TENSOR,
    torch.arange: HighIROp.ARANGE,
    torch.full: HighIROp.FULL,
    torch.zeros: HighIROp.ZEROS,
    torch.ones: HighIROp.ONES,
    # Basic ops
    "add": HighIROp.ADD,
    torch.add: HighIROp.ADD,
    operator.add: HighIROp.ADD,
    operator.iadd: HighIROp.ADD,  # In-place add (+=) treated as regular add
    "sub": HighIROp.SUB,
    torch.sub: HighIROp.SUB,
    operator.sub: HighIROp.SUB,
    "mul": HighIROp.MUL,
    torch.mul: HighIROp.MUL,
    operator.mul: HighIROp.MUL,
    "div": HighIROp.DIV,
    torch.div: HighIROp.DIV,
    operator.truediv: HighIROp.DIV,
    "neg": HighIROp.NEG,
    torch.neg: HighIROp.NEG,
    operator.neg: HighIROp.NEG,
    # Matrix ops
    torch.mm: HighIROp.MM,
    torch.matmul: HighIROp.MATMUL,
    operator.matmul: HighIROp.MATMUL,
    # Activation functions
    torch.relu: HighIROp.RELU,
    F.relu: HighIROp.RELU,
    F.silu: HighIROp.SILU,
    "silu": HighIROp.SILU,
    F.gelu: HighIROp.GELU,
    "gelu": HighIROp.GELU,
    torch.tanh: HighIROp.TANH,
    "tanh": HighIROp.TANH,
    # Unary math
    torch.cos: HighIROp.COS,
    "cos": HighIROp.COS,
    torch.sin: HighIROp.SIN,
    "sin": HighIROp.SIN,
    torch.exp: HighIROp.EXP,
    torch.sqrt: HighIROp.SQRT,
    torch.rsqrt: HighIROp.RSQRT,
    # Power
    torch.pow: HighIROp.POW,
    "pow": HighIROp.POW,
    # Reductions
    torch.sum: HighIROp.SUM,
    "sum": HighIROp.SUM,
    torch.mean: HighIROp.MEAN,
    "mean": HighIROp.MEAN,
    torch.max: HighIROp.MAX,
    "max": HighIROp.MAX,
    torch.min: HighIROp.MIN,
    "min": HighIROp.MIN,
    torch.argmax: HighIROp.ARGMAX,
    "argmax": HighIROp.ARGMAX,
    torch.cumsum: HighIROp.CUMSUM,
    "cumsum": HighIROp.CUMSUM,
    "repeat_interleave": HighIROp.REPEAT_INTERLEAVE,
    torch.repeat_interleave: HighIROp.REPEAT_INTERLEAVE,
    # MoE ops
    torch.topk: HighIROp.TOPK,
    "topk": HighIROp.TOPK,
    torch.scatter: HighIROp.SCATTER,
    "scatter": HighIROp.SCATTER,
    torch.scatter_add: HighIROp.SCATTER_ADD,
    "scatter_add": HighIROp.SCATTER_ADD,
    torch.gather: HighIROp.GATHER,
    "gather": HighIROp.GATHER,
    torch.any: HighIROp.ANY,
    "any": HighIROp.ANY,
    # Gradient control (no-op for inference)
    torch._C._set_grad_enabled: HighIROp.SET_GRAD_ENABLED,
    # vmap batch operations
    _add_batch_dim: HighIROp.ADD_BATCH_DIM,
    _remove_batch_dim: HighIROp.REMOVE_BATCH_DIM,
    _vmap_increment_nesting: HighIROp.VMAP_INCREMENT_NESTING,
    _vmap_decrement_nesting: HighIROp.VMAP_DECREMENT_NESTING,
    # Internal PyTorch ops
    lazy_load_decompositions: HighIROp.LAZY_LOAD_DECOMPOSITIONS,
    _enter_autocast: HighIROp.ENTER_AUTOCAST,
    _exit_autocast: HighIROp.EXIT_AUTOCAST,
    torch._C._log_api_usage_once: HighIROp.LOG_API_USAGE,
    # Softmax
    torch.softmax: HighIROp.SOFTMAX,
    F.softmax: HighIROp.SOFTMAX,
    # Normalization
    F.layer_norm: HighIROp.LAYER_NORM,
    # Linear and embedding
    F.linear: HighIROp.LINEAR,
    F.embedding: HighIROp.EMBEDDING,
    # Shape ops
    "view": HighIROp.VIEW,
    "reshape": HighIROp.RESHAPE,
    torch.reshape: HighIROp.RESHAPE,
    "unsqueeze": HighIROp.UNSQUEEZE,
    torch.unsqueeze: HighIROp.UNSQUEEZE,
    "squeeze": HighIROp.SQUEEZE,
    torch.squeeze: HighIROp.SQUEEZE,
    "transpose": HighIROp.TRANSPOSE,
    torch.transpose: HighIROp.TRANSPOSE,
    "permute": HighIROp.PERMUTE,
    torch.permute: HighIROp.PERMUTE,
    "contiguous": HighIROp.CONTIGUOUS,
    "clone": HighIROp.CLONE,
    torch.clone: HighIROp.CLONE,
    "expand": HighIROp.EXPAND,
    torch.cat: HighIROp.CAT,
    # Indexing
    operator.getitem: HighIROp.GETITEM,
    "select": HighIROp.SELECT,
    torch.select: HighIROp.SELECT,
    "slice": HighIROp.SLICE,
    torch.index_select: HighIROp.INDEX,
    # Comparisons
    torch.eq: HighIROp.EQ,
    operator.eq: HighIROp.EQ,
    "eq": HighIROp.EQ,
    torch.ne: HighIROp.NE,
    operator.ne: HighIROp.NE,
    "ne": HighIROp.NE,
    torch.lt: HighIROp.LT,
    operator.lt: HighIROp.LT,
    "lt": HighIROp.LT,
    torch.le: HighIROp.LE,
    operator.le: HighIROp.LE,
    "le": HighIROp.LE,
    torch.gt: HighIROp.GT,
    operator.gt: HighIROp.GT,
    "gt": HighIROp.GT,
    torch.ge: HighIROp.GE,
    operator.ge: HighIROp.GE,
    "ge": HighIROp.GE,
    # Masking
    torch.where: HighIROp.WHERE,
    "masked_fill": HighIROp.MASKED_FILL,
    torch.triu: HighIROp.TRIU,
    # Dropout (usually no-op at inference)
    F.dropout: HighIROp.DROPOUT,
    # Attention
    F.scaled_dot_product_attention: HighIROp.SCALED_DOT_PRODUCT_ATTENTION,
    # Dtype casting methods
    "float": HighIROp.CAST,  # x.float() -> cast to float32
    "half": HighIROp.CAST,   # x.half() -> cast to float16
    "int": HighIROp.CAST,    # x.int() -> cast to int32
    "long": HighIROp.CAST,   # x.long() -> cast to int64
    "bool": HighIROp.CAST,   # x.bool() -> cast to bool
    # Scalar extraction
    "item": HighIROp.ITEM,   # x.item() -> extract scalar from single-element tensor
    # Control flow
    "to": HighIROp.MOVE_TO,
    "output": HighIROp.OUTPUT,
}

high_ir_op_to_high_ir_node: dict[HighIROp, type[HighIRNode]] = {
    # Existing ops
    HighIROp.CREATE_TENSOR: HighIRCreateTensor,
    HighIROp.PLACEHOLDER: HighIRPlaceholder,
    HighIROp.GETATTR: HighIRGetattr,
    HighIROp.ADD: HighIRAdd,
    HighIROp.RELU: HighIRRelu,
    HighIROp.MOVE_TO: HighIRMoveTo,
    HighIROp.OUTPUT: HighIROutput,
    HighIROp.FUSED_ADD_RELU: HighIRFusedAddRelu,
    HighIROp.FUSED_MUL_SILU: HighIRFusedMulSilu,
    HighIROp.FUSED_ADD_SILU: HighIRFusedAddSilu,
    HighIROp.FUSED_ADD_GELU: HighIRFusedAddGelu,
    HighIROp.MUL: HighIRMul,
    HighIROp.MM: HighIRMM,
    # Basic arithmetic
    HighIROp.SUB: HighIRSub,
    HighIROp.DIV: HighIRDiv,
    HighIROp.NEG: HighIRNeg,
    # Matrix ops
    HighIROp.MATMUL: HighIRMatmul,
    # Activation functions
    HighIROp.SILU: HighIRSilu,
    HighIROp.GELU: HighIRGelu,
    HighIROp.TANH: HighIRTanh,
    # Unary math
    HighIROp.COS: HighIRCos,
    HighIROp.SIN: HighIRSin,
    HighIROp.EXP: HighIRExp,
    HighIROp.SQRT: HighIRSqrt,
    HighIROp.RSQRT: HighIRRsqrt,
    HighIROp.POW: HighIRPow,
    # Reductions
    HighIROp.SUM: HighIRSum,
    HighIROp.MEAN: HighIRMean,
    HighIROp.MAX: HighIRMax,
    HighIROp.MIN: HighIRMin,
    HighIROp.ARGMAX: HighIRArgmax,
    HighIROp.CUMSUM: HighIRCumsum,
    HighIROp.REPEAT_INTERLEAVE: HighIRRepeatInterleave,
    HighIROp.SET_GRAD_ENABLED: HighIRSetGradEnabled,
    # Softmax
    HighIROp.SOFTMAX: HighIRSoftmax,
    # Normalization
    HighIROp.LAYER_NORM: HighIRLayerNorm,
    # Linear and embedding
    HighIROp.LINEAR: HighIRLinear,
    HighIROp.EMBEDDING: HighIREmbedding,
    # Shape ops
    HighIROp.VIEW: HighIRView,
    HighIROp.RESHAPE: HighIRReshape,
    HighIROp.UNSQUEEZE: HighIRUnsqueeze,
    HighIROp.SQUEEZE: HighIRSqueeze,
    HighIROp.TRANSPOSE: HighIRTranspose,
    HighIROp.PERMUTE: HighIRPermute,
    HighIROp.CONTIGUOUS: HighIRContiguous,
    HighIROp.CLONE: HighIRClone,
    HighIROp.EXPAND: HighIRExpand,
    HighIROp.CAT: HighIRCat,
    # Tensor creation
    HighIROp.ARANGE: HighIRArange,
    HighIROp.FULL: HighIRFull,
    HighIROp.ZEROS: HighIRZeros,
    HighIROp.ONES: HighIROnes,
    # Indexing
    HighIROp.GETITEM: HighIRGetitem,
    HighIROp.SELECT: HighIRSelect,
    HighIROp.SLICE: HighIRSlice,
    HighIROp.INDEX: HighIRIndex,
    # Comparisons
    HighIROp.EQ: HighIREq,
    HighIROp.NE: HighIRNe,
    HighIROp.LT: HighIRLt,
    HighIROp.LE: HighIRLe,
    HighIROp.GT: HighIRGt,
    HighIROp.GE: HighIRGe,
    # Masking
    HighIROp.WHERE: HighIRWhere,
    HighIROp.MASKED_FILL: HighIRMaskedFill,
    HighIROp.TRIU: HighIRTriu,
    # Dropout
    HighIROp.DROPOUT: HighIRDropout,
    # Attention
    HighIROp.SCALED_DOT_PRODUCT_ATTENTION: HighIRScaledDotProductAttention,
    # Casting
    HighIROp.CAST: HighIRCast,
    # Scalar extraction
    HighIROp.ITEM: HighIRItem,
    # MoE ops
    HighIROp.TOPK: HighIRTopk,
    HighIROp.SCATTER: HighIRScatter,
    HighIROp.SCATTER_ADD: HighIRScatterAdd,
    HighIROp.GATHER: HighIRGather,
    HighIROp.ANY: HighIRAny,
    # vmap batch operations
    HighIROp.ADD_BATCH_DIM: HighIRAddBatchDim,
    HighIROp.REMOVE_BATCH_DIM: HighIRRemoveBatchDim,
    HighIROp.VMAP_INCREMENT_NESTING: HighIRVmapIncrementNesting,
    HighIROp.VMAP_DECREMENT_NESTING: HighIRVmapDecrementNesting,
    # Internal PyTorch ops
    HighIROp.LAZY_LOAD_DECOMPOSITIONS: HighIRLazyLoadDecompositions,
    HighIROp.ENTER_AUTOCAST: HighIREnterAutocast,
    HighIROp.EXIT_AUTOCAST: HighIRExitAutocast,
    HighIROp.LOG_API_USAGE: HighIRLogApiUsage,
}

high_ir_compiler_passes: list[CompilerPass[HighIRNode]] = [
    CompilerPass(
        transforms=[
            Transform(
                pattern=[
                    Pattern("ir_op", HighIROp.ADD),
                    Pattern("ir_op", HighIROp.RELU),
                ],
                output=HighIROp.FUSED_ADD_RELU,
            ),
            Transform(
                pattern=[
                    Pattern("ir_op", HighIROp.ADD),
                    Pattern("ir_op", HighIROp.SILU),
                ],
                output=HighIROp.FUSED_ADD_SILU,
            ),
            Transform(
                pattern=[
                    Pattern("ir_op", HighIROp.ADD),
                    Pattern("ir_op", HighIROp.GELU),
                ],
                output=HighIROp.FUSED_ADD_GELU,
            ),
            # GLU pattern: silu(gate) * up - fuses SILU followed by MUL
            Transform(
                pattern=[
                    Pattern("ir_op", HighIROp.SILU),
                    Pattern("ir_op", HighIROp.MUL),
                ],
                output=HighIROp.FUSED_MUL_SILU,
            ),
        ]
    ),
]


def get_high_ir_node(fx_op, fx_node: torch.fx.Node) -> Optional[HighIRNode]:
    ir_op = fx_op_to_high_ir_op.get(fx_op)
    if not ir_op:
        return None
    ir_node_type = high_ir_op_to_high_ir_node.get(ir_op)
    if ir_node_type:
        ir_node = ir_node_type(
            fx_node=fx_node, value_id=fx_node.name, inputs=fx_node.all_input_nodes
        )
    return ir_node


def fx_to_high_ir(gm: torch.fx.GraphModule) -> list[HighIRNode]:
    ir_graph: list[HighIRNode] = []
    for i, node in enumerate(gm.graph.nodes):
        ir_node = None

        # Handle FX opcodes first
        if node.op == "placeholder":
            # Input tensors and model parameters
            ir_node = HighIRPlaceholder(
                fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes)
            )
        elif node.op == "get_attr":
            # Accessing module attributes
            ir_node = HighIRGetattr(
                fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes)
            )
        elif node.op == "output":
            # Return value
            ir_node = HighIROutput(
                fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes)
            )
        elif node.op in ("call_function", "call_method"):
            # Special handling for "to" method - can be device transfer or dtype cast
            if node.target == "to" and len(node.args) >= 2:
                target_arg = node.args[1]
                if isinstance(target_arg, torch.dtype):
                    # Dtype casting with explicit dtype
                    ir_node = HighIRCast(
                        fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes),
                        cast_method="cast"  # explicit .to(dtype) uses "cast"
                    )
                else:
                    # Device transfer
                    ir_node = HighIRMoveTo(
                        fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes)
                    )
            elif node.target in ("float", "half", "int", "long", "bool"):
                # Dtype casting methods (e.g., x.float(), x.half())
                ir_node = HighIRCast(
                    fx_node=node, value_id=node.name, inputs=list(node.all_input_nodes),
                    cast_method=node.target  # preserve original method name
                )
            else:
                # Function or method calls - look up by target
                ir_node = get_high_ir_node(fx_op=node.target, fx_node=node)
            if not ir_node:
                # Try source_fn_stack as fallback
                source_fn_stack = node.meta.get("source_fn_stack")
                if source_fn_stack and len(source_fn_stack) > 0:
                    source_fn_stack = source_fn_stack[0]
                    if source_fn_stack and len(source_fn_stack) > 0:
                        node_key = source_fn_stack[0]
                        if node_key:
                            ir_node = get_high_ir_node(fx_op=node_key, fx_node=node)
        elif node.op == "call_module":
            # Submodule calls - typically decomposed before reaching here
            raise Exception(f"call_module not supported: {node.target}")

        if ir_node:
            ir_graph.append(ir_node)
        else:
            debug(f"Unsupported FX op: {node.op} / {node.target}. ir_graph: {ir_graph}")
            raise Exception(f"Unsupported FX op: {node.op} / {node.target}")
    return ir_graph


def high_ir_print_tabular(nodes: List[HighIRNode]) -> None:
    if nodes is None or len(nodes) == 0:
        debug("IR Nodes list is empty")
        return None

    # took most of the code from PyTorch torch/fx/graph.py
    try:
        from tabulate import tabulate
    except ImportError:
        debug(
            "`print_tabular` relies on the library `tabulate`, "
            "which could not be found on this machine. Run `pip "
            "install tabulate` to install the library."
        )
        raise

    node_specs = [
        [
            n.ir_op,
            n.value_id,
            n.inputs,
            n.fx_node.args,
            n.fx_node.kwargs,
        ]
        for n in nodes
    ]
    debug(
        tabulate(
            node_specs,
            headers=[
                "opcode",
                "value_id",
                "inputs",
                "args",
                "kwargs",
            ],
        )
    )
