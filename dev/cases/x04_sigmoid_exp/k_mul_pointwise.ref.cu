extern "C" __global__
__launch_bounds__(256) void k_mul_pointwise(const float* y, const float* x, float* mul) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 171) {
        int a0 = _gid;
        float in0__u0 = x[a0 * 3];
        float in1__u0 = y[a0 * 3];
        float v2__u0 = (1.0f / (1.0f + expf(0.0f - in0__u0))) * expf(in1__u0);
        float in0__u1 = x[a0 * 3 + 1];
        float in1__u1 = y[a0 * 3 + 1];
        float v2__u1 = (1.0f / (1.0f + expf(0.0f - in0__u1))) * expf(in1__u1);
        float in0__u2 = x[a0 * 3 + 2];
        float in1__u2 = y[a0 * 3 + 2];
        float v2__u2 = (1.0f / (1.0f + expf(0.0f - in0__u2))) * expf(in1__u2);
        mul[a0 * 3] = v2__u0;
        mul[a0 * 3 + 1] = v2__u1;
        mul[a0 * 3 + 2] = v2__u2;
    }
}
