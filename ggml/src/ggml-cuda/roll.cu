#include "ggml-cuda/common.cuh"
#include "roll.cuh"

static __forceinline__ __device__ int64_t wrap_index(const int64_t idx, const int64_t ne) {
    if (idx < 0) {
        return idx + ne;
    }
    if (idx >= ne) {
        return idx - ne;
    }
    return idx;
}

template <typename T>
static __global__ void roll(const T * __restrict__ src,
                                T * __restrict__ dst,
                                const int64_t ne00,
                                const int64_t ne01,
                                const int64_t ne02,
                                const int64_t ne03,
                                const int     s0,
                                const int     s1,
                                const int     s2,
                                const int     s3) {
    const int64_t idx        = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;
    const int64_t n_elements = ne00 * ne01 * ne02 * ne03;

    if (idx >= n_elements) {
        return;
    }

    const int64_t i0 = idx % ne00;
    const int64_t i1 = (idx / ne00) % ne01;
    const int64_t i2 = (idx / (ne00 * ne01)) % ne02;
    const int64_t i3 = (idx / (ne00 * ne01 * ne02)) % ne03;

    const int64_t d0 = wrap_index(i0 - s0, ne00);
    const int64_t d1 = wrap_index(i1 - s1, ne01);
    const int64_t d2 = wrap_index(i2 - s2, ne02);
    const int64_t d3 = wrap_index(i3 - s3, ne03);

    dst[i3 * (ne00 * ne01 * ne02) + i2 * (ne01 * ne00) + i1 * ne00 + i0] =
        src[d3 * (ne00 * ne01 * ne02) + d2 * (ne01 * ne00) + d1 * ne00 + d0];
}

template<typename dst_t>
static void roll_cuda_float(
    const dst_t * src, dst_t * dst,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int s0, const int s1, const int s2, const int s3, cudaStream_t stream) {

    int64_t sz         = (ne00 * ne01 * ne02 * ne03);
    int64_t num_blocks = (sz + CUDA_ROLL_BLOCK_SIZE - 1) / CUDA_ROLL_BLOCK_SIZE;

    roll<<<num_blocks, CUDA_ROLL_BLOCK_SIZE, 0, stream>>>(
        src, dst, ne00, ne01, ne02, ne03, s0, s1, s2, s3);
}

static void roll_cuda(
    const void * src, void * dst, ggml_type dst_type,
    const int ne00, const int ne01, const int ne02, const int ne03,
    const int s0, const int s1, const int s2, const int s3, cudaStream_t stream) {

    switch (dst_type) {
        case GGML_TYPE_F32:
            roll_cuda_float((float *) src, (float *) dst, ne00, ne01, ne02, ne03, s0, s1, s2, s3, stream);
            break;
        case GGML_TYPE_F16:
            roll_cuda_float((half *) src, (half *) dst, ne00, ne01, ne02, ne03, s0, s1, s2, s3, stream);
            break;
        case GGML_TYPE_BF16:
            roll_cuda_float((__nv_bfloat16 *) src, (__nv_bfloat16 *) dst, ne00, ne01, ne02, ne03, s0, s1, s2, s3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

void ggml_cuda_op_roll(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    int s0 = dst->op_params[0];
    int s1 = dst->op_params[1];
    int s2 = dst->op_params[2];
    int s3 = dst->op_params[3];

    const ggml_tensor * src0   = dst->src[0];
    // const float *       src0_d = (const float *) dst->src[0]->data;
    // float *             dst_d  = (float *) dst->data;

    GGML_TENSOR_UNARY_OP_LOCALS;

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32 || dst->src[0]->type == GGML_TYPE_F16 || dst->src[0]->type == GGML_TYPE_BF16);
    GGML_ASSERT(ggml_are_same_shape(dst->src[0], dst));

    cudaStream_t stream = ctx.stream();

    roll_cuda(src0->data, dst->data, dst->type, ne00, ne01, ne02, ne03, s0, s1, s2, s3, stream);
}
