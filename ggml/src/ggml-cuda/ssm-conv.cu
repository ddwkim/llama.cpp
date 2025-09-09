#include "ssm-conv.cuh"

template <size_t split_d_inner>
static __global__ void ssm_conv_f32_d3(const float * __restrict__ src0, const float * __restrict__ src1,
                                           const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                           float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                           const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float w0 = w_block[tid * stride_w + 0];
    const float w1 = w_block[tid * stride_w + 1];
    const float w2 = w_block[tid * stride_w + 2];

    const float * x_ch = x_block + tid * stride_x;
    float r0 = x_ch[0], r1 = x_ch[1], r2 = x_ch[2];

    float * y_ptr = y_block + tid;

#pragma unroll
    for (int64_t i = 0; i < n_t; ++i) {
        float sumf = r0 * w0 + r1 * w1 + r2 * w2;

        *y_ptr = sumf;
        y_ptr += stride_y;        

        if (i + 3 < n_t + 3 - 1) {
            const float newv = __ldg(&x_ch[i + 3]);
            r0 = r1; r1 = r2; r2 = newv;
        }
    }
}

template <size_t split_d_inner>
static __global__ void ssm_conv_f32_d4(const float * __restrict__ src0, const float * __restrict__ src1,
                                           const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                           float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                           const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float w0 = w_block[tid * stride_w + 0];
    const float w1 = w_block[tid * stride_w + 1];
    const float w2 = w_block[tid * stride_w + 2];
    const float w3 = w_block[tid * stride_w + 3];

    const float * x_ch = x_block + tid * stride_x;
    float r0 = x_ch[0], r1 = x_ch[1], r2 = x_ch[2], r3 = x_ch[3];

    float * y_ptr = y_block + tid;

#pragma unroll
    for (int64_t i = 0; i < n_t; ++i) {
        float sumf = r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3;

        *y_ptr = sumf;
        y_ptr += stride_y;        

        if (i + 4 < n_t + 4 - 1) {
            const float newv = __ldg(&x_ch[i + 4]);
            r0 = r1; r1 = r2; r2 = r3; r3 = newv;
        }
    }
}

template <size_t split_d_inner, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32_d3(const float * __restrict__ src0, const float * __restrict__ src1,
                                                      const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                                      float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                                      const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float w0 = w_block[tid * stride_w + 0];
    const float w1 = w_block[tid * stride_w + 1];
    const float w2 = w_block[tid * stride_w + 2];

    const float * x_ch = x_block + tid * stride_x;
    float r0 = x_ch[0], r1 = x_ch[1], r2 = x_ch[2];

    float * y_ptr = y_block + tid;

#pragma unroll
    for (int64_t i = 0; i < split_n_t; i++) {
        if (bidz * split_n_t + i < n_t) {
            float sumf = r0 * w0 + r1 * w1 + r2 * w2;

            *y_ptr = sumf;
            y_ptr += stride_y;        

            if (i + 3 < n_t + 3 - 1) {
                const float newv = __ldg(&x_ch[i + 3]);
                r0 = r1; r1 = r2; r2 = newv;
            }
        }
    }
}

template <size_t split_d_inner, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32_d4(const float * __restrict__ src0, const float * __restrict__ src1,
                                                      const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                                      float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                                      const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const float w0 = w_block[tid * stride_w + 0];
    const float w1 = w_block[tid * stride_w + 1];
    const float w2 = w_block[tid * stride_w + 2];
    const float w3 = w_block[tid * stride_w + 3];

    const float * x_ch = x_block + tid * stride_x;
    float r0 = x_ch[0], r1 = x_ch[1], r2 = x_ch[2], r3 = x_ch[3];

    float * y_ptr = y_block + tid;

#pragma unroll
    for (int64_t i = 0; i < split_n_t; i++) {
        if (bidz * split_n_t + i < n_t) {
            float sumf = r0 * w0 + r1 * w1 + r2 * w2 + r3 * w3;

            *y_ptr = sumf;
            y_ptr += stride_y;        

            if (i + 4 < n_t + 4 - 1) {
                const float newv = __ldg(&x_ch[i + 4]);
                r0 = r1; r1 = r2; r2 = r3; r3 = newv;
            }
        }
    }
}


static void ssm_conv_f32_cuda(const float * src0, const float * src1, const int src0_nb0, const int src0_nb1,
                                  const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                  const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                                  const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    if (n_t <= 32) {
        const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
        if (nc == 4) {
            ssm_conv_f32_d4<threads><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                          dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else if (nc == 3) {
            ssm_conv_f32_d3<threads><<<blocks, threads, 0, stream>>>(src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                          dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            GGML_ABORT("Only support kernel size = 3 or size = 4 right now.");
        }
    } else {
        int64_t split_n_t;
        if (n_t < 512)          split_n_t = 32;
        else if (n_t < 2048)    split_n_t = 64;
        else                    split_n_t = 128;

        dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
        
        if (nc == 4) {
            switch (split_n_t) {
                case 32:
                    ssm_conv_long_token_f32_d4<threads, 32><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
                case 64:
                    ssm_conv_long_token_f32_d4<threads, 64><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
                case 128:
                    ssm_conv_long_token_f32_d4<threads, 128><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
            }
        } else if (nc == 3) {
            switch (split_n_t) {
                case 32:
                    ssm_conv_long_token_f32_d3<threads, 32><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
                case 64:
                    ssm_conv_long_token_f32_d3<threads, 64><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
                case 128:
                    ssm_conv_long_token_f32_d3<threads, 128><<<blocks, threads, 0, stream>>>(
                        src0, src1, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
                    break;
            }
        } else {
            GGML_ABORT("Only support kernel size = 3 or size = 4 right now.");
        }
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = dst->ne[1];                 // tokens per sequence
    const int64_t n_s = dst->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    float *       dst_d  = (float *) dst->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    ssm_conv_f32_cuda(src0_d, src1_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, dst->nb[0], dst->nb[1],
                      dst->nb[2], nc, nr, n_t, n_s, stream);
}
