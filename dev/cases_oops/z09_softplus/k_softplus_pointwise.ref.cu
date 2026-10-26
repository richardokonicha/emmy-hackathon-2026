extern "C" __global__
__launch_bounds__(256) void k_softplus_pointwise(const float* softplus_one, const float* softplus_zero, const float* x, float* softplus) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 128) {
        int a0 = _gid;
        float in2__u0 = x[a0 * 2];
        float in1__u0 = softplus_zero[0];
        float in0__u0 = softplus_one[0];
        float v6__u0 = fmaxf(in1__u0, in2__u0) + log((in0__u0 + expf((0.0f - fabsf(in2__u0)))));
        float in2__u1 = x[a0 * 2 + 1];
        float in1__u1 = softplus_zero[0];
        float in0__u1 = softplus_one[0];
        float v6__u1 = fmaxf(in1__u1, in2__u1) + log((in0__u1 + expf((0.0f - fabsf(in2__u1)))));
        float2 _vs_v6__u0_292124d7 = make_float2(v6__u0, v6__u1);
        *reinterpret_cast<float2*>(&softplus[a0 * 2]) = _vs_v6__u0_292124d7;
    }
}
