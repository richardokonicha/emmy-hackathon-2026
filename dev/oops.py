"""Op-coverage variants: exercises every scalar op family emmy 0.3.11 can lower."""

import math

import numpy as np

_erf = np.vectorize(math.erf)


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    dict(id="z01_tanh",
         code="torch.tanh(torch.randn(300))",
         ref=lambda x: [np.tanh(x.astype(np.float64)).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="z02_gelu",
         code="import torch.nn.functional as F; F.gelu(torch.randn(300))",
         ref=lambda x: [(0.5 * x * (1.0 + _erf(x / 2 ** 0.5))).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="z03_gelu_tanh",
         code="import torch.nn.functional as F; F.gelu(torch.randn(300), approximate='tanh')",
         ref=lambda x: [(0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x ** 3)))).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="z05_logexp",
         code="torch.log(torch.exp(torch.randn(300) * 2.0))",
         ref=lambda x: [x * np.float32(2.0)], rtol=1e-5, atol=1e-6),
    dict(id="z06_sign",
         code="torch.sign(torch.randn(128))",
         ref=lambda x: [np.sign(x)], rtol=0, atol=0),
    dict(id="z07_sin_cos",
         code="torch.sin(torch.randn(128)) * torch.cos(torch.randn(128))",
         ref=lambda a, b: [(np.sin(a.astype(np.float64)) * np.cos(b.astype(np.float64))).astype(np.float32)],
         rtol=1e-4, atol=1e-4),
    dict(id="z08_pow",
         code="torch.pow(torch.abs(torch.randn(64)) + 0.5, 3.0)",
         ref=lambda x: [(np.abs(x) + 0.5) ** 3], rtol=1e-5, atol=1e-5),
    dict(id="z09_softplus",
         code="torch.nn.functional.softplus(torch.randn(256))",
         ref=lambda x: [(np.log1p(np.exp(-np.abs(x.astype(np.float64)))) * 0 + np.log1p(np.exp(x.astype(np.float64)))).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
    dict(id="z10_rsqrt",
         code="torch.rsqrt(torch.abs(torch.randn(256)) + 0.25)",
         ref=lambda x: [(1.0 / np.sqrt(np.abs(x) + 0.25)).astype(np.float32)], rtol=1e-5, atol=1e-6),
    dict(id="z11_tanh_sigmoid",
         code="torch.tanh(torch.randn(200)) * torch.sigmoid(torch.randn(200))",
         ref=lambda a, b: [(np.tanh(a.astype(np.float64)) * (1.0 / (1.0 + np.exp(-b.astype(np.float64))))).astype(np.float32)],
         rtol=1e-5, atol=1e-6),
]
