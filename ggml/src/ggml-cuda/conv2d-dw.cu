#include "conv2d-dw.cuh"

struct conv_params {
    int in_w, in_h;
    int out_w, out_h;
    int kernel_w, kernel_h;
    int stride_x, stride_y;
    int padding_x, padding_y;
    int dilation_x, dilation_y;
    int channels, batches;
};

struct kernel_bounds {
    int y_min, y_max;
    int x_min, x_max;
};

__device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int out_x, int out_y, const conv_params & params) {
    kernel_bounds bounds;
    bounds.y_min = max(0, (params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.y_max =
        min(params.kernel_h,
            (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.x_min = max(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    bounds.x_max =
        min(params.kernel_w,
            (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    return bounds;
}

__device__ __forceinline__ int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
    return out_coord * stride + kern_coord * dilation - padding;
}

struct whcn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + c * params.in_w * params.in_h + y * params.in_w + x;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return c * params.kernel_h * params.kernel_w + ky * params.kernel_w + kx;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h +
               y * params.out_w + x;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
                                          int & out_x) {
        out_x = global_idx % params.out_w;
        out_y = (global_idx / params.out_w) % params.out_h;
        c     = (global_idx / (params.out_w * params.out_h)) % params.channels;
        n     = global_idx / (params.out_w * params.out_h * params.channels);
    }
};

struct cwhn_layout {
    __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.in_w * params.in_h) + (y * params.in_w + x) * params.channels + c;
    }

    __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
        return (ky * params.kernel_w + kx) * params.channels + c;
    }

    __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
        return n * (params.channels * params.out_w * params.out_h) + y * (params.out_w * params.channels) +
               x * params.channels + c;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
                                          int & out_x) {
        c     = global_idx % params.channels;
        out_x = (global_idx / params.channels) % params.out_w;
        out_y = (global_idx / (params.channels * params.out_w)) % params.out_h;
        n     = global_idx / (params.channels * params.out_w * params.out_h);
    }
};

template <typename T, typename Layout>
__global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
                                 const int in_w, const int in_h, const int out_w, const int out_h,
                                 const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
                                 const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
                                 const int channels, const int batches) {
    const int global_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batches * channels * out_h * out_w;
    if (global_idx >= total_elements) return;

    conv_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                           stride_x, stride_y, padding_x, padding_y,
                           dilation_x, dilation_y, channels, batches };

    int n, c, oy, ox;
    Layout::unpack_indices(global_idx, params, n, c, oy, ox);

    T acc = 0;

    const int in_row_start = oy * params.stride_y - params.padding_y;
    const int in_col_start = ox * params.stride_x - params.padding_x;
    const int in_row_end   = in_row_start + (params.kernel_h - 1) * params.dilation_y + 1;
    const int in_col_end   = in_col_start + (params.kernel_w - 1) * params.dilation_x + 1;

    const bool fully_inside =
        (in_row_start >= 0) && (in_col_start >= 0) &&
        (in_row_end   <= params.in_h) && (in_col_end   <= params.in_w);

    unsigned warp_mask = __activemask();
    const bool warp_all_inside = __all_sync(warp_mask, fully_inside);

    if (warp_all_inside) {
        // ---- FAST PATH: No boundary check, fixed size/fully converged ----
        #pragma unroll
        for (int ky = 0; ky < params.kernel_h; ++ky) {
            const int iy = in_row_start + ky * params.dilation_y;
            #pragma unroll
            for (int kx = 0; kx < params.kernel_w; ++kx) {
                const int ix = in_col_start + kx * params.dilation_x;
                const T ival = input[Layout::input_index(n, c, iy, ix, params)];
                const T wval = kernel[Layout::kernel_index(c, ky, kx, params)];
                acc += ival * wval;
            }
        }
    } else {
        // ---- SLOW PATH: Boundary check ----
        #pragma unroll
        for (int ky = 0; ky < params.kernel_h; ++ky) {
            const int iy = in_row_start + ky * params.dilation_y;
            const bool row_valid = (unsigned)iy < (unsigned)params.in_h;

            #pragma unroll
            for (int kx = 0; kx < params.kernel_w; ++kx) {
                const int ix = in_col_start + kx * params.dilation_x;
                const bool valid = row_valid && ((unsigned)ix < (unsigned)params.in_w);

                T ival = 0;
                if (valid) {
                    ival = input[Layout::input_index(n, c, iy, ix, params)];
                }
                const T wval = kernel[Layout::kernel_index(c, ky, kx, params)];
                acc += ival * wval;
            }
        }
    }

    output[Layout::output_index(n, c, oy, ox, params)] = acc;
}


// Tiled version for stride = 1 (better performance)
template <typename T, typename Layout, int TILE_X, int TILE_Y>
__global__ void conv2d_dw_kernel_tiled(
    const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* __restrict__ output,
    int in_w, int in_h,
    int out_w, int out_h,
    int kernel_w, int kernel_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y,
    int channels, int batches)
{
    const int cz = blockIdx.z;
    const int c  = cz % channels;
    const int n  = cz / channels;
    if (n >= batches) return;

    const int ox0 = blockIdx.x * TILE_X;
    const int oy0 = blockIdx.y * TILE_Y;

    const int in_x0 = ox0 * stride_x - padding_x;
    const int in_y0 = oy0 * stride_y - padding_y;

    const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
    const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;

    extern __shared__ T smem[]; // tile_w * tile_h + kernel_w * kernel_h
    T* tile = smem;
    T* kernel_tile = smem + tile_w * tile_h;

    conv_params params = {
        in_w, in_h, out_w, out_h, kernel_w, kernel_h,
        stride_x, stride_y, padding_x, padding_y,
        dilation_x, dilation_y, channels, batches
    };

    // Load input tile to shared memory
    for (int dy = threadIdx.y; dy < tile_h; dy += blockDim.y) {
        const int gy = in_y0 + dy;
        const bool y_ok = (unsigned)gy < (unsigned)in_h;

        for (int dx = threadIdx.x; dx < tile_w; dx += blockDim.x) {
            const int gx = in_x0 + dx;
            T v = 0;
            if (y_ok && (unsigned)gx < (unsigned)in_w) {
                v = input[Layout::input_index(n, c, gy, gx, params)];
            }
            tile[dy * tile_w + dx] = v;
        }
    }
    
    // Load kernel to shared memory
    for (int ky = threadIdx.y; ky < kernel_h; ky += blockDim.y) {
        for (int kx = threadIdx.x; kx < kernel_w; kx += blockDim.x) {
            kernel_tile[ky * kernel_w + kx] = kernel[Layout::kernel_index(c, ky, kx, params)];
        }
    }
    __syncthreads();

    const int ox = ox0 + threadIdx.x;
    const int oy = oy0 + threadIdx.y;

    if (ox < out_w && oy < out_h) {
        T acc = 0;

        const int sx0 = threadIdx.x * stride_x;
        const int sy0 = threadIdx.y * stride_y;

        #pragma unroll
        for (int ky = 0; ky < kernel_h; ++ky) {
            const int sy = sy0 + ky * dilation_y;
            #pragma unroll
            for (int kx = 0; kx < kernel_w; ++kx) {
                const int sx = sx0 + kx * dilation_x;
                const T a = tile[sy * tile_w + sx];    
                const T w = kernel_tile[ky * kernel_w + kx]; 
                acc = fmaf(a, w, acc);
            }
        }

        output[Layout::output_index(n, c, oy, ox, params)] = acc;
    }
}

// ===== COMPILE-TIME TEMPLATE KERNEL =====
template<int KW, int KH, int TX, int TY, typename T, typename Layout>
__global__ void conv2d_dw_kernel_tiled_ct(
    const T* __restrict__ input,
    const T* __restrict__ kernel,
    T* __restrict__ output,
    int in_w, int in_h,
    int out_w, int out_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y,
    int channels, int batches)
{
    const int cz = blockIdx.z;
    const int c  = cz % channels;
    const int n  = cz / channels;
    if (n >= batches) return;

    const int ox0 = blockIdx.x * TX;
    const int oy0 = blockIdx.y * TY;

    const int in_x0 = ox0 * stride_x - padding_x;
    const int in_y0 = oy0 * stride_y - padding_y;

    const int tile_w = (TX - 1) * stride_x + (KW - 1) * dilation_x + 1;
    const int tile_h = (TY - 1) * stride_y + (KH - 1) * dilation_y + 1;

    extern __shared__ T smem[]; // tile_w * tile_h + KW * KH
    T* tile = smem;
    T* kernel_tile = smem + tile_w * tile_h;

    conv_params params = {
        in_w, in_h, out_w, out_h, KW, KH,
        stride_x, stride_y, padding_x, padding_y,
        dilation_x, dilation_y, channels, batches
    };

    // Load input tile to shared memory
    for (int dy = threadIdx.y; dy < tile_h; dy += blockDim.y) {
        const int gy = in_y0 + dy;
        const bool y_ok = (unsigned)gy < (unsigned)in_h;

        for (int dx = threadIdx.x; dx < tile_w; dx += blockDim.x) {
            const int gx = in_x0 + dx;
            T v = 0;
            if (y_ok && (unsigned)gx < (unsigned)in_w) {
                v = input[Layout::input_index(n, c, gy, gx, params)];
            }
            tile[dy * tile_w + dx] = v;
        }
    }
    
    // Load kernel to shared memory
    for (int ky = threadIdx.y; ky < KH; ky += blockDim.y) {
        for (int kx = threadIdx.x; kx < KW; kx += blockDim.x) {
            kernel_tile[ky * KW + kx] = kernel[Layout::kernel_index(c, ky, kx, params)];
        }
    }
    __syncthreads();

    const int ox = ox0 + threadIdx.x;
    const int oy = oy0 + threadIdx.y;

    if (ox < out_w && oy < out_h) {
        T acc = 0;

        const int sx0 = threadIdx.x * stride_x;
        const int sy0 = threadIdx.y * stride_y;

        #pragma unroll
        for (int ky = 0; ky < KH; ++ky) {
            const int sy = sy0 + ky * dilation_y;
            #pragma unroll
            for (int kx = 0; kx < KW; ++kx) {
                const int sx = sy * tile_w + (sx0 + kx * dilation_x);
                const T a = tile[sx];
                const T w = kernel_tile[ky * KW + kx];
                acc = fmaf(a, w, acc);
            }
        }

        output[Layout::output_index(n, c, oy, ox, params)] = acc;
    }
}

// ===== COMPILE-TIME LAUNCHER =====
template<int TX, int TY>
void launch_tiled_stride1_ct(
    int K,
    const float* __restrict__ input,
    const float* __restrict__ kernel,
    float* __restrict__ output,
    int in_w, int in_h,
    int out_w, int out_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y,
    int channels, int batches,
    dim3 grid, dim3 block, size_t smem, cudaStream_t stream)
{
    switch (K) {
        case 3: 
            conv2d_dw_kernel_tiled_ct<3,3,TX,TY,float,whcn_layout><<<grid, block, smem, stream>>>(
                input, kernel, output,
                in_w, in_h, out_w, out_h,
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            break;
        case 5: 
            conv2d_dw_kernel_tiled_ct<5,5,TX,TY,float,whcn_layout><<<grid, block, smem, stream>>>(
                input, kernel, output,
                in_w, in_h, out_w, out_h,
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            break;
        default: 
            // Fallback to general tiled version for other kernel sizes
            conv2d_dw_kernel_tiled<float,whcn_layout,TX,TY><<<grid, block, smem, stream>>>(
                input, kernel, output,
                in_w, in_h, out_w, out_h,
                K, K,  // kernel_w, kernel_h
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            break;
    }
}


void ggml_cuda_op_conv2d_dw(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F32 && input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);
    const float * w_d = (const float *) kernel->data;
    const float * x_d = (const float *) input->data;
    float *       y_d = (float *) dst->data;

    const int32_t * p          = (const int32_t *) dst->op_params;
    const int       stride_x   = p[0];
    const int       stride_y   = p[1];
    const int       padding_x  = p[2];
    const int       padding_y  = p[3];
    const int       dilation_x = p[4];
    const int       dilation_y = p[5];

    const int in_w     = input->ne[0];
    const int in_h     = input->ne[1];
    const int kernel_w = kernel->ne[0];
    const int kernel_h = kernel->ne[1];
    const int out_w    = dst->ne[0];
    const int out_h    = dst->ne[1];
    const int channels = dst->ne[2];
    const int batches  = dst->ne[3];

    cudaStream_t st = ctx.stream();

    // Choose kernel based on stride
    if (stride_x == 1 && stride_y == 1 && kernel_w <= 7 && kernel_h <= 7) {
        // Use tiled kernel for stride = 1 (better performance)
        const int TILE_X = 32;
        const int TILE_Y = 8;
        
        // Calculate tile dimensions
        const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
        const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;
        const size_t shared_mem_size = (tile_w * tile_h + kernel_w * kernel_h) * sizeof(float);
        
        // Check if shared memory size is reasonable (fallback to regular kernel if too large)
        // Most CUDA devices support 48KB shared memory per block, but we'll be conservative
        const size_t max_shared_mem = 32 * 1024; // 32KB limit
        
        if (shared_mem_size <= max_shared_mem) {
            // 2D block configuration for tiled kernel
            dim3 blocks_2d((out_w + TILE_X - 1) / TILE_X, 
                            (out_h + TILE_Y - 1) / TILE_Y, 
                            batches * channels);
            dim3 threads_2d(TILE_X, TILE_Y);
            
            // Choose between compile-time template and regular tiled kernel
            bool use_compile_time = (kernel_w == kernel_h && (kernel_w == 3 || kernel_w == 5));
            
            if (ggml_is_contiguous(input)) {
                if (use_compile_time) {
                    launch_tiled_stride1_ct<TILE_X, TILE_Y>(
                        kernel_w,
                        x_d, w_d, y_d,
                        in_w, in_h, out_w, out_h,
                        stride_x, stride_y,
                        padding_x, padding_y,
                        dilation_x, dilation_y,
                        channels, batches,
                        blocks_2d, threads_2d, shared_mem_size, st
                    );
                } else {
                    conv2d_dw_kernel_tiled<float, whcn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size, st>>>(
                        x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                        dilation_x, dilation_y, channels, batches);
                }
            } else if (ggml_is_contiguous_channels(input)) {
                // For cwhn_layout, we don't have compile-time template support yet, so use regular tiled
                conv2d_dw_kernel_tiled<float, cwhn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size, st>>>(
                    x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                    dilation_x, dilation_y, channels, batches);
            } else {
                GGML_ABORT("Unsupported memory layout for conv_2d_dw");
            }
        } else {
            // Fallback to regular kernel if shared memory is too large
            const int total  = batches * channels * out_h * out_w;
            const int blocks = (total + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;

            if (ggml_is_contiguous(input)) {
                conv2d_dw_kernel<float, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
                    x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                    dilation_x, dilation_y, channels, batches);
            } else if (ggml_is_contiguous_channels(input)) {
                conv2d_dw_kernel<float, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
                    x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                    dilation_x, dilation_y, channels, batches);
            } else {
                GGML_ABORT("Unsupported memory layout for conv_2d_dw");
            }
        }
    } else {
        // Use regular kernel for stride > 1
        const int total  = batches * channels * out_h * out_w;
        const int blocks = (total + CUDA_CONV2D_DW_BLOCK_SIZE - 1) / CUDA_CONV2D_DW_BLOCK_SIZE;

        if (ggml_is_contiguous(input)) {
            conv2d_dw_kernel<float, whcn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
                x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                dilation_x, dilation_y, channels, batches);
        } else if (ggml_is_contiguous_channels(input)) {
            conv2d_dw_kernel<float, cwhn_layout><<<blocks, CUDA_CONV2D_DW_BLOCK_SIZE, 0, st>>>(
                x_d, w_d, y_d, in_w, in_h, out_w, out_h, kernel_w, kernel_h, stride_x, stride_y, padding_x, padding_y,
                dilation_x, dilation_y, channels, batches);
        } else {
            GGML_ABORT("Unsupported memory layout for conv_2d_dw");
        }
    }
}
