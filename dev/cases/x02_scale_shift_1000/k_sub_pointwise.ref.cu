extern "C" __global__
__launch_bounds__(256) void k_sub_pointwise(const float* sub_c1, const float* mul_c1, const float* x, float* sub) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 500) {
        int a0 = _gid;
        float in0__u0 = mul_c1[0];
        float in2__u0 = x[a0 * 2];
        float in1__u0 = sub_c1[0];
        float v1__u0 = (in0__u0 * in2__u0) - in1__u0;
        float in0__u1 = mul_c1[0];
        float in2__u1 = x[a0 * 2 + 1];
        float in1__u1 = sub_c1[0];
        float v1__u1 = (in0__u1 * in2__u1) - in1__u1;
        float2 _vs_v1__u0_48f3a952 = make_float2(v1__u0, v1__u1);
        *reinterpret_cast<float2*>(&sub[a0 * 2]) = _vs_v1__u0_48f3a952;
    }
}
