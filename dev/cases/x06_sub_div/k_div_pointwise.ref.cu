extern "C" __global__
__launch_bounds__(256) void k_div_pointwise(const float* add_c1, const float* y, const float* x, const float* z, float* div) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 341) {
        int a0 = _gid;
        float in1__u0 = x[a0 * 3];
        float in2__u0 = y[a0 * 3];
        float in0__u0 = add_c1[0];
        float in3__u0 = z[a0 * 3];
        float v2__u0 = (in1__u0 - in2__u0) / (in0__u0 + in3__u0);
        float in1__u1 = x[a0 * 3 + 1];
        float in2__u1 = y[a0 * 3 + 1];
        float in0__u1 = add_c1[0];
        float in3__u1 = z[a0 * 3 + 1];
        float v2__u1 = (in1__u1 - in2__u1) / (in0__u1 + in3__u1);
        float in1__u2 = x[a0 * 3 + 2];
        float in2__u2 = y[a0 * 3 + 2];
        float in0__u2 = add_c1[0];
        float in3__u2 = z[a0 * 3 + 2];
        float v2__u2 = (in1__u2 - in2__u2) / (in0__u2 + in3__u2);
        div[a0 * 3] = v2__u0;
        div[a0 * 3 + 1] = v2__u1;
        div[a0 * 3 + 2] = v2__u2;
    }
}
