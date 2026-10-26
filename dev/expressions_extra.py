"""Hidden-set-like variants used to harden compile.py. Not part of the submission."""

import numpy as np

REDUCE_KNOBS = "WORK=t128,REDUCE=coop"


def _softmax(x, axis=-1):
    m = x.max(axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / e.sum(axis=axis, keepdims=True)


OPEN = [
    dict(id="x01_pointwise_odd",
         code="torch.randn(257) + torch.randn(257)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="x02_scale_shift_1000",
         code="torch.randn(1000) * 2.5 - 3.0",
         ref=lambda x: [x * np.float32(2.5) - np.float32(3.0)], rtol=1e-6, atol=1e-6),
    dict(id="x03_relu_mul_777",
         code="x = torch.randn(777); y = torch.randn(777); torch.relu(x) * y",
         ref=lambda x, y: [np.maximum(x, 0) * y], rtol=0, atol=0),
    dict(id="x04_sigmoid_exp",
         code="x = torch.randn(513); y = torch.randn(513); torch.sigmoid(x) * torch.exp(y)",
         ref=lambda x, y: [(1.0 / (1.0 + np.exp(-x.astype(np.float64))) * np.exp(y.astype(np.float64))).astype(np.float32)],
         rtol=1e-5, atol=1e-5),
    dict(id="x05_abs_where",
         code="x = torch.randn(300); torch.where(x > 0, x, -x)",
         ref=lambda x: [np.abs(x)], rtol=0, atol=0),
    dict(id="x06_sub_div",
         code="x = torch.randn(1023); y = torch.randn(1023); z = torch.randn(1023); (x - y) / (z + 1.0)",
         ref=lambda x, y, z: [(x - y) / (z + np.float32(1.0))], rtol=1e-5, atol=1e-5),

    dict(id="r05_sum_4097",
         code="torch.randn(4097).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r06_max_10000",
         code="torch.randn(10000).max()",
         ref=lambda x: [x.max()], rtol=0, atol=0, knobs=REDUCE_KNOBS),
    dict(id="r07_mean_12345",
         code="torch.randn(12345).mean()",
         ref=lambda x: [x.astype(np.float64).mean().astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r08_row_sum_33x777",
         code="torch.randn(33, 777).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="r09_row_max_17x333",
         code="torch.randn(17, 333).max(-1)",
         ref=lambda x: [x.max(-1)], rtol=0, atol=0, knobs=REDUCE_KNOBS),

    dict(id="m03_matmul_64x128_128x32",
         code="torch.randn(64, 128) @ torch.randn(128, 32)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="m04_matmul_100x50_50x75",
         code="torch.randn(100, 50) @ torch.randn(50, 75)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="m05_matmul_256x256_256x256",
         code="torch.randn(256, 256) @ torch.randn(256, 256)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),
    dict(id="m06_matmul_33x65_65x17",
         code="torch.randn(33, 65) @ torch.randn(65, 17)",
         ref=lambda a, b: [(a.astype(np.float64) @ b.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-3),

    dict(id="f04_rmsnorm_16x777",
         code="x = torch.randn(16, 777); w = torch.randn(777); "
              "x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-5) * w",
         ref=lambda x, w: [(x / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + 1e-5) * w).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f05_softmax_16x333",
         code="torch.softmax(torch.randn(16, 333), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="f06_l2norm_16x511",
         code="x = torch.randn(16, 511); x / torch.sqrt((x * x).sum(-1, keepdim=True))",
         ref=lambda x: [(x / np.sqrt((x.astype(np.float64) ** 2).sum(-1, keepdims=True))).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="f07_layernorm_32x256",
         code="x = torch.randn(32, 256); w = torch.randn(256); b = torch.randn(256); "
              "(x - x.mean(-1, keepdim=True)) / torch.sqrt(((x - x.mean(-1, keepdim=True)) ** 2).mean(-1, keepdim=True) + 1e-5) * w + b",
         ref=lambda x, w, b: [((x - x.astype(np.float64).mean(-1, keepdims=True))
                               / np.sqrt(((x - x.astype(np.float64).mean(-1, keepdims=True)) ** 2).mean(-1, keepdims=True) + 1e-5)
                               * w + b).astype(np.float32)],
         rtol=1e-3, atol=1e-4),

    dict(id="a03_attention_32x48",
         code="q = torch.randn(32, 48); kt = torch.randn(48, 32); v = torch.randn(32, 48); "
              "torch.softmax(q @ kt / 6.9, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 6.9)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="a04_attention_128x64",
         code="q = torch.randn(128, 64); kt = torch.randn(64, 128); v = torch.randn(128, 64); "
              "torch.softmax(q @ kt / 8.0, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 8.0)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="a05_attention_17x33",
         code="q = torch.randn(17, 33); kt = torch.randn(33, 17); v = torch.randn(17, 33); "
              "torch.softmax(q @ kt / 5.74, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 5.74)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),

    dict(id="x07_pointwise_big",
         code="torch.randn(65537) + torch.randn(65537)",
         ref=lambda a, b: [a + b], rtol=0, atol=0),
    dict(id="r11_sum_65537",
         code="torch.randn(65537).sum()",
         ref=lambda x: [x.astype(np.float64).sum().astype(np.float32)], rtol=1e-3, atol=1e-3, knobs=REDUCE_KNOBS),
    dict(id="r10_row_sum_9x7",
         code="torch.randn(9, 7).sum(-1)",
         ref=lambda x: [x.astype(np.float64).sum(-1).astype(np.float32)], rtol=1e-4, atol=1e-4, knobs=REDUCE_KNOBS),
    dict(id="f08_softmax_9x17",
         code="torch.softmax(torch.randn(9, 17), dim=-1)",
         ref=lambda x: [_softmax(x.astype(np.float64)).astype(np.float32)], rtol=1e-3, atol=1e-5),
    dict(id="a06_attention_8x8",
         code="q = torch.randn(8, 8); kt = torch.randn(8, 8); v = torch.randn(8, 8); "
              "torch.softmax(q @ kt / 2.828427, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 2.828427)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
    dict(id="a07_attention_9x17",
         code="q = torch.randn(9, 17); kt = torch.randn(17, 9); v = torch.randn(9, 17); "
              "torch.softmax(q @ kt / 4.1231056, dim=-1) @ v",
         ref=lambda q, kt, v: [(_softmax((q.astype(np.float64) @ kt.astype(np.float64)) / 4.1231056)
                                @ v.astype(np.float64)).astype(np.float32)],
         rtol=1e-3, atol=1e-4),
]
