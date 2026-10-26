extern "C" __global__
__launch_bounds__(256) void k_gelu_pointwise(const float* gelu_inv_sqrt2, const float* gelu_one, const float* gelu_half, const float* x, float* gelu) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 150) {
        int a0 = _gid;
        float in1__u0 = gelu_inv_sqrt2[0];
        float in3__u0 = x[a0 * 2];
        float in2__u0 = gelu_one[0];
        float in0__u0 = gelu_half[0];
        float v4__u0 = (in2__u0 + erff((in1__u0 * in3__u0))) * (in0__u0 * in3__u0);
        float in1__u1 = gelu_inv_sqrt2[0];
        float in3__u1 = x[a0 * 2 + 1];
        float in2__u1 = gelu_one[0];
        float in0__u1 = gelu_half[0];
        float v4__u1 = (in2__u1 + erff((in1__u1 * in3__u1))) * (in0__u1 * in3__u1);
        float2 _vs_v4__u0_80f5490e = make_float2(v4__u0, v4__u1);
        *reinterpret_cast<float2*>(&gelu[a0 * 2]) = _vs_v4__u0_80f5490e;
    }
}
