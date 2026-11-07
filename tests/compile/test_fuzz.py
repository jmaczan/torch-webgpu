"""Differential fuzzing of the Inductor backend: seeded random programs
(pointwise math, broadcasting, reductions, views, concatenation, dtype
round trips) are compiled for webgpu and compared with CPU eager.

These cover fusion combinations OpInfo samples never produce: several
reductions feeding pointwise epilogues, strided / transposed / sliced reads,
broadcasts of reduced values, mixed dtypes inside one kernel.

TORCH_WEBGPU_FUZZ_PROGRAMS sets the number of programs (default 60);
TORCH_WEBGPU_FUZZ_SEED offsets the seeds. A failure prints the program.
"""

import os
import random

import pytest
import torch

N_PROGRAMS = int(os.environ.get("TORCH_WEBGPU_FUZZ_PROGRAMS", "60"))
SEED0 = int(os.environ.get("TORCH_WEBGPU_FUZZ_SEED", "0"))

UNARY = {
    "exp": lambda x: torch.exp(x.clamp(-10, 10)),
    "tanh": torch.tanh,
    "sin": torch.sin,
    "abs": torch.abs,
    "neg": torch.neg,
    "sigmoid": torch.sigmoid,
    "relu": torch.relu,
    "sqrt": lambda x: torch.sqrt(x.abs()),
    "log1p": lambda x: torch.log1p(x.abs()),
    "rsqrt": lambda x: torch.rsqrt(x.abs() + 1),
    "square": lambda x: x * x,
    "gelu": torch.nn.functional.gelu,
    "floor": torch.floor,
    "erf": torch.erf,
}

BINARY = {
    "add": torch.add,
    "sub": torch.sub,
    "mul": torch.mul,
    "div": lambda a, b: a / (b.abs() + 0.5),
    "maximum": torch.maximum,
    "minimum": torch.minimum,
    "where": lambda a, b: torch.where(a > b, a, b * 0.5),
    "atan2": torch.atan2,
    "pow": lambda a, b: (a.abs() + 0.1).pow(b.clamp(-2, 2)),
}

REDUCE = {
    "sum": lambda x, d: x.sum(d, keepdim=True),
    "mean": lambda x, d: x.mean(d, keepdim=True),
    "amax": lambda x, d: x.amax(d, keepdim=True),
    "amin": lambda x, d: x.amin(d, keepdim=True),
    "logsumexp": lambda x, d: torch.logsumexp(x, d, keepdim=True),
    "var": lambda x, d: x.var(d, keepdim=True, unbiased=False),
    "softmax": lambda x, d: torch.softmax(x, d),
    "argmax": lambda x, d: x.argmax(d, keepdim=True).float(),
}


class Program:
    """A random straight-line program over a pool of same-shape values."""

    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        r = self.rng
        rank = r.choice([1, 2, 2, 3, 3])
        self.shape = tuple(r.choice([1, 3, 7, 16, 33, 64, 129]) for _ in range(rank))
        self.n_inputs = r.randint(1, 3)
        self.steps = []
        pool = self.n_inputs
        for _ in range(r.randint(3, 12)):
            kind = r.choices(["unary", "binary", "reduce", "view", "cast", "cat"], [4, 4, 3, 2, 1, 1])[0]
            a = r.randrange(pool)
            if kind == "unary":
                self.steps.append(("unary", r.choice(list(UNARY)), a))
            elif kind == "binary":
                self.steps.append(("binary", r.choice(list(BINARY)), a, r.randrange(pool)))
            elif kind == "reduce":
                self.steps.append(("reduce", r.choice(list(REDUCE)), a, r.randrange(rank)))
            elif kind == "view":
                self.steps.append(("view", r.choice(["flip", "transpose", "slice", "roll", "contiguous_t"]), a,
                                   r.randrange(rank)))
            elif kind == "cast":
                self.steps.append(("cast", r.choice(["half", "bfloat16", "int", "bool"]), a))
            else:
                self.steps.append(("cat", None, a, r.randrange(pool), r.randrange(rank)))
            pool += 1
        self.n_outputs = r.randint(1, 2)

    def __repr__(self):
        lines = [f"shape={self.shape} inputs={self.n_inputs}"]
        lines += [f"  v{self.n_inputs + i} = {s}" for i, s in enumerate(self.steps)]
        return "\n".join(lines)

    def inputs(self):
        g = torch.Generator().manual_seed(self.rng.randint(0, 2**31))
        return [torch.randn(self.shape, generator=g) for _ in range(self.n_inputs)]

    def __call__(self, *xs):
        vals = list(xs)
        for step in self.steps:
            kind, name, a = step[:3]
            x = vals[a]
            if kind == "unary":
                y = UNARY[name](x)
            elif kind == "binary":
                y = BINARY[name](x, vals[step[3]])
            elif kind == "reduce":
                # Reduced values broadcast back: later steps mix them with
                # full-size values (normalization-style epilogues).
                y = REDUCE[name](x, step[3]).expand_as(x) * 1.0 + x * 0.5
            elif kind == "view":
                d = step[3]
                if name == "flip":
                    y = x.flip(d)
                elif name == "transpose":
                    y = x.transpose(0, -1).contiguous().transpose(0, -1)
                elif name == "slice":
                    n = x.shape[d]
                    y = torch.cat([x.narrow(d, n // 2, n - n // 2), x.narrow(d, 0, n // 2)], d)
                elif name == "roll":
                    y = x.roll(1, d)
                else:
                    y = x.mT.contiguous().mT * 1.0 if x.dim() >= 2 else x * 1.0
            elif kind == "cast":
                if name == "half":
                    y = x.half().float()
                elif name == "bfloat16":
                    y = x.bfloat16().float()
                elif name == "int":
                    y = (x * 4).to(torch.int32).float() * 0.25
                else:
                    y = (x > 0).float() - (x < -1).float()
            else:
                b, d = vals[step[3]], step[4]
                n = x.shape[d]
                y = torch.cat([x.narrow(d, 0, n - n // 2), b.narrow(d, 0, n // 2)], d)
            # Keep magnitudes bounded so tolerances stay meaningful.
            vals.append(y.clamp(-1e3, 1e3))
        return tuple(vals[-self.n_outputs:])


@pytest.mark.parametrize("seed", range(SEED0, SEED0 + N_PROGRAMS))
def test_random_program(seed):
    p = Program(seed)
    xs = p.inputs()
    expected = p(*xs)
    torch._dynamo.reset()
    actual = torch.compile(p, dynamic=False)(*[x.to("webgpu") for x in xs])
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), e, atol=2e-3, rtol=2e-3, equal_nan=True, msg=lambda m: f"{m}\n{p}")
