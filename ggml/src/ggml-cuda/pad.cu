#include "pad.cuh"

template<typename T>
static __global__ void pad(const T* src, T * dst,
                               const int lp0, const int rp0, const int lp1, const int rp1,
                               const int lp2, const int rp2, const int lp3, const int rp3,
                               const int ne0, const int ne1, const int ne2, const int ne3) {
    // blockIdx.z: i3*ne2+i2
    // blockIdx.y: i1
    // blockIDx.x: i0 / CUDA_PAD_BLOCK_SIZE
    // gridDim.y:  ne1
    int i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int i1 = blockIdx.y;
    int i2 = blockIdx.z % ne2;
    int i3 = blockIdx.z / ne2;
    if (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        return;
    }

    // operation
    const int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    if ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        const int64_t i00 = i0 - lp0;
        const int64_t i01 = i1 - lp1;
        const int64_t i02 = i2 - lp2;
        const int64_t i03 = i3 - lp3;
        const int64_t ne02 = ne2 - lp2 - rp2;
        const int64_t ne01 = ne1 - lp1 - rp1;
        const int64_t ne00 = ne0 - lp0 - rp0;

        const int64_t src_idx = i03*(ne00*ne01*ne02) + i02*(ne00*ne01) + i01*ne00 + i00;

        dst[dst_idx] = src[src_idx];
    } else {
        dst[dst_idx] = 0.0f;
    }
}

template<typename T>
static void pad_cuda(const T * src, T * dst,
    const int lp0, const int rp0, const int lp1, const int rp1,
    const int lp2, const int rp2, const int lp3, const int rp3,
    const int ne0, const int ne1, const int ne2, const int ne3, cudaStream_t stream) {
    int num_blocks = (ne0 + CUDA_PAD_BLOCK_SIZE - 1) / CUDA_PAD_BLOCK_SIZE;
    dim3 gridDim(num_blocks, ne1, ne2*ne3);
    pad<T><<<gridDim, CUDA_PAD_BLOCK_SIZE, 0, stream>>>(src, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3, ne0, ne1, ne2, ne3);
}

void ggml_cuda_op_pad(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int32_t lp0 = ((const int32_t*)(dst->op_params))[0];
    const int32_t rp0 = ((const int32_t*)(dst->op_params))[1];
    const int32_t lp1 = ((const int32_t*)(dst->op_params))[2];
    const int32_t rp1 = ((const int32_t*)(dst->op_params))[3];
    const int32_t lp2 = ((const int32_t*)(dst->op_params))[4];
    const int32_t rp2 = ((const int32_t*)(dst->op_params))[5];
    const int32_t lp3 = ((const int32_t*)(dst->op_params))[6];
    const int32_t rp3 = ((const int32_t*)(dst->op_params))[7];

    switch (dst->type) {
        case GGML_TYPE_F32: {
            const float * src0_f32 = (const float *)src0->data;
            float * dst_f32 = (float *)dst->data;
            pad_cuda<float>(src0_f32, dst_f32,
                lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
            break;
        }
        case GGML_TYPE_F16: {
            const half * src0_f16 = (const half *)src0->data;
            half * dst_f16 = (half *)dst->data;
            pad_cuda<half>(src0_f16, dst_f16,
                lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
            break;
        }
        case GGML_TYPE_BF16: {
            const __nv_bfloat16 * src0_bf16 = (const __nv_bfloat16 *)src0->data;
            __nv_bfloat16 * dst_bf16 = (__nv_bfloat16 *)dst->data;
            pad_cuda<__nv_bfloat16>(src0_bf16, dst_bf16,
                lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
            break;
        }
        default:
            GGML_ABORT("unsupported type\n");
    }
    
}