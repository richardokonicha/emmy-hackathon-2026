extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* x1, const float* x0, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 100) {
        int a0 = _gid;
        float in0__u0 = x0[a0 * 2];
        float in1__u0 = x1[a0 * 2];
        float v2__u0 = tanhf(in0__u0) * (1.0f / (1.0f + expf(0.0f - in1__u0)));
        float in0__u1 = x0[a0 * 2 + 1];
        float in1__u1 = x1[a0 * 2 + 1];
        float v2__u1 = tanhf(in0__u1) * (1.0f / (1.0f + expf(0.0f - in1__u1)));
        float2 _vs_v2__u0_0d6d9abc = make_float2(v2__u0, v2__u1);
        *reinterpret_cast<float2*>(&mul[a0 * 2]) = _vs_v2__u0_0d6d9abc;
    }
}
