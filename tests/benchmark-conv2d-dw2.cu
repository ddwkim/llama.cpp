#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>
#include <random>
#include <vector>
#include <cstdio>
#include <cstdlib>

// NVTX helper functions
inline nvtxRangeId_t nvtx_push(const char* name, uint32_t argb=0xFF1f77b4) {
    nvtxEventAttributes_t a{};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = argb;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = name;
    return nvtxRangeStartEx(&a);
}

inline void nvtx_pop(nvtxRangeId_t id) {
    nvtxRangeEnd(id);
}

// Nsight Compute profiling repeat macro
#ifndef NCU_REPEAT
#define NCU_REPEAT 1
#endif

// CUDA error checking macro
#define BENCHMARK_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ===== BRANCH RATIO CALCULATION =====
// Calculate branch ratio analytically instead of using atomic counters
struct BranchRatio {
    float fully_inside_ratio;
    float boundary_ratio;
};

BranchRatio calculate_branch_ratio_analytical(
    int in_w, int in_h, int out_w, int out_h,
    int kernel_w, int kernel_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y
) {
    // Calculate how many output positions are fully inside vs boundary
    int fully_inside_count = 0;
    int total_count = 0;
    
    for (int out_y = 0; out_y < out_h; out_y++) {
        for (int out_x = 0; out_x < out_w; out_x++) {
            total_count++;
            
            // Calculate input bounds for this output position
            int in_row_start = out_y * stride_y - padding_y;
            int in_col_start = out_x * stride_x - padding_x;
            int in_row_end = in_row_start + (kernel_h - 1) * dilation_y + 1;
            int in_col_end = in_col_start + (kernel_w - 1) * dilation_x + 1;
            
            // Check if fully inside
            bool fully_inside = 
                (in_row_start >= 0) && (in_col_start >= 0) &&
                (in_row_end <= in_h) && (in_col_end <= in_w);
                
            if (fully_inside) {
                fully_inside_count++;
            }
        }
    }
    
    BranchRatio ratio;
    ratio.fully_inside_ratio = (float)fully_inside_count / total_count * 100.0f;
    ratio.boundary_ratio = (float)(total_count - fully_inside_count) / total_count * 100.0f;
    
    return ratio;
}

// Calculate and print branch ratio (will be defined after g_log_file)
void print_branch_ratio(const char* kernel_name, const BranchRatio& ratio);

// ===== CURRENT KERNEL IMPLEMENTATION =====
// Copy the necessary structures and kernel from conv2d-dw.cu
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
    bounds.y_max = min(params.kernel_h, (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
    bounds.x_min = max(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
    bounds.x_max = min(params.kernel_w, (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
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
        return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h + y * params.out_w + x;
    }

    __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y, int & out_x) {
        out_x = global_idx % params.out_w;
        out_y = (global_idx / params.out_w) % params.out_h;
        c     = (global_idx / (params.out_w * params.out_h)) % params.channels;
        n     = global_idx / (params.out_w * params.out_h * params.channels);
    }
};


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



// ===== BACKUP KERNEL IMPLEMENTATION =====
// Copy from conv2d-dw-backup.cu (uncommented version)
template <typename T, typename Layout>
__global__ void conv2d_dw_kernel_backup(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
                                        const int in_w, const int in_h, const int out_w, const int out_h,
                                        const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
                                        const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
                                        const int channels, const int batches) {
    const int global_idx     = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_elements = batches * channels * out_h * out_w;

    if (global_idx >= total_elements) {
        return;
    }

    conv_params params = { in_w,     in_h,      out_w,     out_h,      kernel_w,   kernel_h, stride_x,
                           stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches };

    int batch_idx, channel_idx, out_y_idx, out_x_idx;
    Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);

    T accumulator = 0;
    
    kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);

    for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
        int in_y_idx = calculate_input_coord(out_y_idx, kern_y, params.stride_y, params.dilation_y, params.padding_y);

        for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
            int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);

            const T input_val  = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
            const T kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];

            accumulator += input_val * kernel_val;
        }
    }

    output[Layout::output_index(batch_idx, channel_idx, out_y_idx, out_x_idx, params)] = accumulator;
}

// Global log file pointer for benchmark function
FILE* g_log_file = nullptr;

// Calculate and print branch ratio
void print_branch_ratio(const char* kernel_name, const BranchRatio& ratio) {
    printf("  %s: Fully inside: %.1f%%, Boundary: %.1f%%\n", 
           kernel_name, ratio.fully_inside_ratio, ratio.boundary_ratio);
    
    if (g_log_file) {
        fprintf(g_log_file, "  %s: Fully inside: %.1f%%, Boundary: %.1f%%\n", 
                kernel_name, ratio.fully_inside_ratio, ratio.boundary_ratio);
    }
}

// Benchmark conv2d-dw kernel (backup implementation)
void benchmark_conv2d_dw_backup(
    int in_w, int in_h, int channels, int batches,
    int kernel_w, int kernel_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y,
    int iterations
) {
    // Calculate output dimensions
    int out_w = (in_w + 2 * padding_x - dilation_x * (kernel_w - 1) - 1) / stride_x + 1;
    int out_h = (in_h + 2 * padding_y - dilation_y * (kernel_h - 1) - 1) / stride_y + 1;

    // Calculate memory sizes
    size_t input_size = (size_t)batches * channels * in_h * in_w * sizeof(float);
    size_t kernel_size = (size_t)channels * kernel_h * kernel_w * sizeof(float);
    size_t output_size = (size_t)batches * channels * out_h * out_w * sizeof(float);

    // Allocate GPU memory
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_input, input_size));
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_output, output_size));

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> h_input(batches * channels * in_h * in_w);
    std::vector<float> h_kernel(channels * kernel_h * kernel_w);

    for (auto& val : h_input) val = dis(gen);
    for (auto& val : h_kernel) val = dis(gen);

    BENCHMARK_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    BENCHMARK_CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_size, cudaMemcpyHostToDevice));

    // Create CUDA stream
    cudaStream_t stream;
    BENCHMARK_CUDA_CHECK(cudaStreamCreate(&stream));

    // Calculate launch configuration
    int threads_per_block = 256;
    int total_elements = batches * channels * out_h * out_w;
    int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

    // NVTX range for kernel execution
    auto r = nvtx_push("bench_backup", 0xFF1f77b4);
    
    // Warm-up run
    conv2d_dw_kernel_backup<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
        d_input, d_kernel, d_output,
        in_w, in_h, out_w, out_h,
        kernel_w, kernel_h,
        stride_x, stride_y,
        padding_x, padding_y,
        dilation_x, dilation_y,
        channels, batches
    );
    BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    
    // Benchmark runs
    std::vector<float> durations;
    durations.reserve(iterations);
    
    for (int i = 0; i < iterations; ++i) {
        cudaEvent_t start, stop;
        BENCHMARK_CUDA_CHECK(cudaEventCreate(&start));
        BENCHMARK_CUDA_CHECK(cudaEventCreate(&stop));
        
        BENCHMARK_CUDA_CHECK(cudaEventRecord(start, stream));
        
        conv2d_dw_kernel_backup<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
            d_input, d_kernel, d_output,
            in_w, in_h, out_w, out_h,
            kernel_w, kernel_h,
            stride_x, stride_y,
            padding_x, padding_y,
            dilation_x, dilation_y,
            channels, batches
        );
        
        BENCHMARK_CUDA_CHECK(cudaEventRecord(stop, stream));
        BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        float duration;
        BENCHMARK_CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
        durations.push_back(duration);
        
        BENCHMARK_CUDA_CHECK(cudaEventDestroy(start));
        BENCHMARK_CUDA_CHECK(cudaEventDestroy(stop));
    }
    
    // Calculate statistics
    float total_time = 0.0f;
    float min_time = durations[0];
    float max_time = durations[0];
    
    for (float duration : durations) {
        total_time += duration;
        min_time = std::min(min_time, duration);
        max_time = std::max(max_time, duration);
    }
    
    float avg_time = total_time / iterations;
    
    printf("conv2d_dw_kernel_backup results (%d iterations):\n", iterations);
    printf("  Average time: %.3f ms\n", avg_time);
    printf("  Min time: %.3f ms\n", min_time);
    printf("  Max time: %.3f ms\n", max_time);
    printf("  Total time: %.3f ms\n\n", total_time);
    
    nvtx_pop(r);

    // Cleanup
    BENCHMARK_CUDA_CHECK(cudaStreamDestroy(stream));
    BENCHMARK_CUDA_CHECK(cudaFree(d_input));
    BENCHMARK_CUDA_CHECK(cudaFree(d_kernel));
    BENCHMARK_CUDA_CHECK(cudaFree(d_output));
}


// Unified benchmark function for conv2d-dw kernel
void benchmark_conv2d_dw(
    int in_w, int in_h, int channels, int batches,
    int kernel_w, int kernel_h,
    int stride_x, int stride_y,
    int padding_x, int padding_y,
    int dilation_x, int dilation_y,
    int iterations
) {
    // Calculate output dimensions
    int out_w = (in_w + 2 * padding_x - dilation_x * (kernel_w - 1) - 1) / stride_x + 1;
    int out_h = (in_h + 2 * padding_y - dilation_y * (kernel_h - 1) - 1) / stride_y + 1;

    // Calculate memory sizes
    size_t input_size = (size_t)batches * channels * in_h * in_w * sizeof(float);
    size_t kernel_size = (size_t)channels * kernel_h * kernel_w * sizeof(float);
    size_t output_size = (size_t)batches * channels * out_h * out_w * sizeof(float);

    // Allocate GPU memory
    float *d_input = nullptr, *d_kernel = nullptr, *d_output = nullptr;
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_input, input_size));
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_kernel, kernel_size));
    BENCHMARK_CUDA_CHECK(cudaMalloc(&d_output, output_size));

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(-1.0f, 1.0f);

    std::vector<float> h_input(batches * channels * in_h * in_w);
    std::vector<float> h_kernel(channels * kernel_h * kernel_w);

    for (auto& val : h_input) val = dis(gen);
    for (auto& val : h_kernel) val = dis(gen);

    BENCHMARK_CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size, cudaMemcpyHostToDevice));
    BENCHMARK_CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kernel_size, cudaMemcpyHostToDevice));

    // Create CUDA stream
    cudaStream_t stream;
    BENCHMARK_CUDA_CHECK(cudaStreamCreate(&stream));

    // Choose kernel based on stride (same logic as conv2d-dw.cu)
    if (stride_x == 1 && stride_y == 1) {
        // Use tiled kernel for stride = 1 (better performance)
        const int TILE_X = 32;
        const int TILE_Y = 8;
        
        // Calculate tile dimensions
        const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
        const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;
        const size_t shared_mem_size = (tile_w * tile_h + kernel_w * kernel_h) * sizeof(float);
        
        // Check if shared memory size is reasonable (fallback to regular kernel if too large)
        const size_t max_shared_mem = 32 * 1024; // 32KB limit
        
        if (shared_mem_size <= max_shared_mem) {
            // 2D block configuration for tiled kernel
            dim3 blocks_2d((out_w + TILE_X - 1) / TILE_X, 
                            (out_h + TILE_Y - 1) / TILE_Y, 
                            batches * channels);
            dim3 threads_2d(TILE_X, TILE_Y);
            
            printf("Using tiled kernel (stride=1, shared_mem=%.1f KB)\n", shared_mem_size / 1024.0f);
            
            // NVTX range for kernel execution
            auto r = nvtx_push("bench_cur", 0xFF1f77b4);
            
            // Warm-up run
            conv2d_dw_kernel_tiled<float, whcn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size, stream>>>(
                d_input, d_kernel, d_output,
                in_w, in_h, out_w, out_h,
                kernel_w, kernel_h,
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Benchmark runs
            std::vector<float> durations;
            durations.reserve(iterations);
            
            for (int i = 0; i < iterations; ++i) {
                cudaEvent_t start, stop;
                BENCHMARK_CUDA_CHECK(cudaEventCreate(&start));
                BENCHMARK_CUDA_CHECK(cudaEventCreate(&stop));
                
                BENCHMARK_CUDA_CHECK(cudaEventRecord(start, stream));
                
                conv2d_dw_kernel_tiled<float, whcn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size, stream>>>(
                    d_input, d_kernel, d_output,
                    in_w, in_h, out_w, out_h,
                    kernel_w, kernel_h,
                    stride_x, stride_y,
                    padding_x, padding_y,
                    dilation_x, dilation_y,
                    channels, batches
                );
                
                BENCHMARK_CUDA_CHECK(cudaEventRecord(stop, stream));
                BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
                
                float duration;
                BENCHMARK_CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
                durations.push_back(duration);
                
                BENCHMARK_CUDA_CHECK(cudaEventDestroy(start));
                BENCHMARK_CUDA_CHECK(cudaEventDestroy(stop));
            }
            
            // Calculate statistics
            float total_time = 0.0f;
            float min_time = durations[0];
            float max_time = durations[0];
            
            for (float duration : durations) {
                total_time += duration;
                min_time = std::min(min_time, duration);
                max_time = std::max(max_time, duration);
            }
            
            float avg_time = total_time / iterations;
            
            printf("conv2d_dw_kernel_tiled results (%d iterations):\n", iterations);
            printf("  Average time: %.3f ms\n", avg_time);
            printf("  Min time: %.3f ms\n", min_time);
            printf("  Max time: %.3f ms\n", max_time);
            printf("  Total time: %.3f ms\n\n", total_time);
            
            nvtx_pop(r);
            
        } else {
            // Fallback to regular kernel if shared memory is too large
            printf("Shared memory too large (%.1f KB > 32 KB), using regular kernel\n", shared_mem_size / 1024.0f);
            
            // Calculate launch configuration for regular kernel
            int threads_per_block = 256;
            int total_elements = batches * channels * out_h * out_w;
            int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
            
            // NVTX range for kernel execution
            auto r = nvtx_push("bench_cur", 0xFF1f77b4);
            
            // Warm-up run
            conv2d_dw_kernel<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
                d_input, d_kernel, d_output,
                in_w, in_h, out_w, out_h,
                kernel_w, kernel_h,
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            
            // Benchmark runs
            std::vector<float> durations;
            durations.reserve(iterations);
            
            for (int i = 0; i < iterations; ++i) {
                cudaEvent_t start, stop;
                BENCHMARK_CUDA_CHECK(cudaEventCreate(&start));
                BENCHMARK_CUDA_CHECK(cudaEventCreate(&stop));
                
                BENCHMARK_CUDA_CHECK(cudaEventRecord(start, stream));
                
                conv2d_dw_kernel<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
                    d_input, d_kernel, d_output,
                    in_w, in_h, out_w, out_h,
                    kernel_w, kernel_h,
                    stride_x, stride_y,
                    padding_x, padding_y,
                    dilation_x, dilation_y,
                    channels, batches
                );
                
                BENCHMARK_CUDA_CHECK(cudaEventRecord(stop, stream));
                BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
                
                float duration;
                BENCHMARK_CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
                durations.push_back(duration);
                
                BENCHMARK_CUDA_CHECK(cudaEventDestroy(start));
                BENCHMARK_CUDA_CHECK(cudaEventDestroy(stop));
            }
            
            // Calculate statistics
            float total_time = 0.0f;
            float min_time = durations[0];
            float max_time = durations[0];
            
            for (float duration : durations) {
                total_time += duration;
                min_time = std::min(min_time, duration);
                max_time = std::max(max_time, duration);
            }
            
            float avg_time = total_time / iterations;
            
            printf("conv2d_dw_kernel (fallback) results (%d iterations):\n", iterations);
            printf("  Average time: %.3f ms\n", avg_time);
            printf("  Min time: %.3f ms\n", min_time);
            printf("  Max time: %.3f ms\n", max_time);
            printf("  Total time: %.3f ms\n\n", total_time);
            
            nvtx_pop(r);
        }
        
    } else {
        // Use regular kernel for stride > 1
        printf("Using regular kernel (stride > 1)\n");
        
        // Calculate launch configuration for regular kernel
        int threads_per_block = 256;
        int total_elements = batches * channels * out_h * out_w;
        int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
        
        // NVTX range for kernel execution
        auto r = nvtx_push("bench_cur", 0xFF1f77b4);
        
        // Warm-up run
        conv2d_dw_kernel<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
            d_input, d_kernel, d_output,
            in_w, in_h, out_w, out_h,
            kernel_w, kernel_h,
            stride_x, stride_y,
            padding_x, padding_y,
            dilation_x, dilation_y,
            channels, batches
        );
        BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        
        // Benchmark runs
        std::vector<float> durations;
        durations.reserve(iterations);
        
        for (int i = 0; i < iterations; ++i) {
            cudaEvent_t start, stop;
            BENCHMARK_CUDA_CHECK(cudaEventCreate(&start));
            BENCHMARK_CUDA_CHECK(cudaEventCreate(&stop));
            
            BENCHMARK_CUDA_CHECK(cudaEventRecord(start, stream));
            
            conv2d_dw_kernel<float, whcn_layout><<<blocks, threads_per_block, 0, stream>>>(
                d_input, d_kernel, d_output,
                in_w, in_h, out_w, out_h,
                kernel_w, kernel_h,
                stride_x, stride_y,
                padding_x, padding_y,
                dilation_x, dilation_y,
                channels, batches
            );
            
            BENCHMARK_CUDA_CHECK(cudaEventRecord(stop, stream));
            BENCHMARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            
            float duration;
            BENCHMARK_CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
            durations.push_back(duration);
            
            BENCHMARK_CUDA_CHECK(cudaEventDestroy(start));
            BENCHMARK_CUDA_CHECK(cudaEventDestroy(stop));
        }
        
        // Calculate statistics
        float total_time = 0.0f;
        float min_time = durations[0];
        float max_time = durations[0];
        
        for (float duration : durations) {
            total_time += duration;
            min_time = std::min(min_time, duration);
            max_time = std::max(max_time, duration);
        }
        
        float avg_time = total_time / iterations;
        
        printf("conv2d_dw_kernel results (%d iterations):\n", iterations);
        printf("  Average time: %.3f ms\n", avg_time);
        printf("  Min time: %.3f ms\n", min_time);
        printf("  Max time: %.3f ms\n", max_time);
        printf("  Total time: %.3f ms\n\n", total_time);
        
        nvtx_pop(r);
    }

    // Cleanup
    BENCHMARK_CUDA_CHECK(cudaStreamDestroy(stream));
    BENCHMARK_CUDA_CHECK(cudaFree(d_input));
    BENCHMARK_CUDA_CHECK(cudaFree(d_kernel));
    BENCHMARK_CUDA_CHECK(cudaFree(d_output));
}


int main(int argc, char* argv[]) {
    // Create log file with timestamp
    time_t now = time(0);
    struct tm* ltm = localtime(&now);
    char log_filename[256];
    sprintf(log_filename, "conv2d_benchmark_comparison_%04d%02d%02d_%02d%02d%02d.log", 
            1900 + ltm->tm_year, 1 + ltm->tm_mon, ltm->tm_mday,
            ltm->tm_hour, ltm->tm_min, ltm->tm_sec);
    
    // Open log file and set global pointer
    g_log_file = fopen(log_filename, "w");
    if (g_log_file == nullptr) {
        printf("Warning: Could not create log file %s\n", log_filename);
    } else {
        printf("Logging results to: %s\n", log_filename);
    }

    printf("CUDA CONV2D-DW Kernel Comparison Benchmark\n");
    printf("==========================================\n\n");
    printf("Log file: %s\n", log_filename);
    printf("Timestamp: %s", ctime(&now));
    printf("\n");

    // Check command line arguments for custom configuration
    if (argc == 7) {
        // Custom configuration mode: ./program input_size kernel_size stride padding dilation iterations
        int input_size = atoi(argv[1]);
        int kernel_size = atoi(argv[2]);
        int stride = atoi(argv[3]);
        int padding = atoi(argv[4]);
        int dilation = atoi(argv[5]);
        int iterations = atoi(argv[6]);
        
        printf("Custom Configuration Mode:\n");
        printf("Input size: %dx%d\n", input_size, input_size);
        printf("Kernel size: %dx%d\n", kernel_size, kernel_size);
        printf("Stride: (%d,%d)\n", stride, stride);
        printf("Padding: (%d,%d)\n", padding, padding);
        printf("Dilation: (%d,%d)\n", dilation, dilation);
        printf("Iterations: %d\n", iterations);
        printf("\n");
        
        // Calculate output dimensions
        int out_size = (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        
        if (out_size <= 0) {
            printf("Error: Invalid configuration - output size would be %d\n", out_size);
            return 1;
        }
        
        printf("Output size: %dx%d\n", out_size, out_size);
        printf("\n");
        
        try {
            // Calculate branch ratio analytically
            BranchRatio current_branch_ratio = calculate_branch_ratio_analytical(
                input_size, input_size, out_size, out_size,
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                dilation, dilation
            );
            
            // Test current implementation
            benchmark_conv2d_dw(
                input_size, input_size, 64, 1,  // channels=64, batches=1
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                dilation, dilation,
                iterations
            );
            
            // Test backup implementation
            benchmark_conv2d_dw_backup(
                input_size, input_size, 64, 1,  // channels=64, batches=1
                kernel_size, kernel_size,
                stride, stride,
                padding, padding,
                dilation, dilation,
                iterations
            );
            
        } catch (...) {
            printf("Failed to benchmark configuration\n");
            return 1;
        }
        
        if (g_log_file) {
            fclose(g_log_file);
            g_log_file = nullptr;
        }
        
        return 0;
    }
    
    // Full benchmark mode (original behavior)
    printf("Full Benchmark Mode (all test cases)\n");
    printf("Usage for custom configuration: %s <input_size> <kernel_size> <stride> <padding> <dilation> <iterations>\n", argv[0]);
    printf("Example: %s 128 5 2 1 1 100\n\n", argv[0]);
    
    // Default iterations for full benchmark mode
    int default_iterations = 100;
    printf("Using default iterations: %d\n\n", default_iterations);
    
    // Test different problem sizes
    int problem_sizes[][4] = {
        {64, 64, 64, 1},      // Small
        {128, 128, 128, 1},   // Medium
    };
    int num_problem_sizes = 2;

    int kernel_sizes[][2] = {
        {3, 3}, {5, 5}, {7, 7}, {9, 9}
    };
    int num_kernel_sizes = 4;

    int stride_combinations[][2] = {
        {1, 1}, {2, 2}, {3, 3}
    };
    int num_stride_combinations = 3;

    int padding_combinations[][2] = {
        {0, 0}, {1, 1}
    };
    int num_padding_combinations = 2;

    // Test different dilation values
    int dilation_combinations[][2] = {
        {1, 1}, {2, 2}
    };
    int num_dilation_combinations = 2;

    printf("Testing configurations:\n");
    printf("- Problem sizes: %d\n", num_problem_sizes);
    printf("- Kernel sizes: %d\n", num_kernel_sizes);
    printf("- Stride combinations: %d\n", num_stride_combinations);
    printf("- Padding combinations: %d\n", num_padding_combinations);
    printf("- Dilation combinations: %d\n", num_dilation_combinations);
    printf("- Total combinations: %d\n", 
           num_problem_sizes * num_kernel_sizes * num_stride_combinations * 
           num_padding_combinations * num_dilation_combinations);
    printf("\n");

    // Benchmark different configurations
    for (int p = 0; p < num_problem_sizes; p++) {
        int in_w = problem_sizes[p][0];
        int in_h = problem_sizes[p][1];
        int channels = problem_sizes[p][2];
        int batches = problem_sizes[p][3];
        
        for (int k = 0; k < num_kernel_sizes; k++) {
            int kernel_w = kernel_sizes[k][0];
            int kernel_h = kernel_sizes[k][1];
            
            for (int s = 0; s < num_stride_combinations; s++) {
                int stride_x = stride_combinations[s][0];
                int stride_y = stride_combinations[s][1];
                
                for (int pad = 0; pad < num_padding_combinations; pad++) {
                    int padding_x = padding_combinations[pad][0];
                    int padding_y = padding_combinations[pad][1];
                    
                    for (int d = 0; d < num_dilation_combinations; d++) {
                        int dilation_x = dilation_combinations[d][0];
                        int dilation_y = dilation_combinations[d][1];
                        
                        // Skip invalid configurations
                        int out_w = (in_w + 2 * padding_x - dilation_x * (kernel_w - 1) - 1) / stride_x + 1;
                        int out_h = (in_h + 2 * padding_y - dilation_y * (kernel_h - 1) - 1) / stride_y + 1;

                        if (out_w <= 0 || out_h <= 0) continue;

                        printf("=== Testing Configuration ===\n");
                        printf("Input: %dx%dx%dx%d, Kernel: %dx%d, Output: %dx%dx%dx%d\n",
                               batches, channels, in_h, in_w, kernel_h, kernel_w, batches, channels, out_h, out_w);
                        printf("Stride: (%d,%d), Padding: (%d,%d), Dilation: (%d,%d)\n",
                               stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y);
                        printf("=====================================\n\n");

                        try {
                            // Calculate branch ratio analytically
                            BranchRatio current_branch_ratio = calculate_branch_ratio_analytical(
                                in_w, in_h, out_w, out_h,
                                kernel_w, kernel_h,
                                stride_x, stride_y,
                                padding_x, padding_y,
                                dilation_x, dilation_y
                            );
                            
                            // Test current implementation
                            benchmark_conv2d_dw(
                                in_w, in_h, channels, batches,
                                kernel_w, kernel_h,
                                stride_x, stride_y,
                                padding_x, padding_y,
                                dilation_x, dilation_y,
                                default_iterations
                            );
                            
                            // Test backup implementation
                            benchmark_conv2d_dw_backup(
                                in_w, in_h, channels, batches,
                                kernel_w, kernel_h,
                                stride_x, stride_y,
                                padding_x, padding_y,
                                dilation_x, dilation_y,
                                default_iterations
                            );
                        } catch (...) {
                            printf("Failed to benchmark configuration\n");
                        }
                    }
                }
            }
        }
    }

    if (g_log_file) {
        fclose(g_log_file);
        g_log_file = nullptr;
    }
    
    return 0;
} 


// #include <cuda_runtime.h>
// #include <nvtx3/nvToolsExt.h>
// #include <random>
// #include <vector>
// #include <cstdio>
// #include <cstdlib>

// // NVTX helper functions
// inline nvtxRangeId_t nvtx_push(const char* name, uint32_t argb=0xFF1f77b4) {
//     nvtxEventAttributes_t a{};
//     a.version = NVTX_VERSION;
//     a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
//     a.colorType = NVTX_COLOR_ARGB;
//     a.color = argb;
//     a.messageType = NVTX_MESSAGE_TYPE_ASCII;
//     a.message.ascii = name;
//     return nvtxRangeStartEx(&a);
// }

// inline void nvtx_pop(nvtxRangeId_t id) {
//     nvtxRangeEnd(id);
// }

// // CUDA error checking macro
// #define BENCHMARK_CUDA_CHECK(call) do { \
//     cudaError_t err = (call); \
//     if (err != cudaSuccess) { \
//         fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
//         exit(1); \
//     } \
// } while(0)

// // ===== BRANCH RATIO CALCULATION =====
// // Calculate branch ratio analytically instead of using atomic counters
// struct BranchRatio {
//     float fully_inside_ratio;
//     float boundary_ratio;
// };

// BranchRatio calculate_branch_ratio_analytical(
//     int in_w, int in_h, int out_w, int out_h,
//     int kernel_w, int kernel_h,
//     int stride_x, int stride_y,
//     int padding_x, int padding_y,
//     int dilation_x, int dilation_y
// ) {
//     // Calculate how many output positions are fully inside vs boundary
//     int fully_inside_count = 0;
//     int total_count = 0;
    
//     for (int out_y = 0; out_y < out_h; out_y++) {
//         for (int out_x = 0; out_x < out_w; out_x++) {
//             total_count++;
            
//             // Calculate input bounds for this output position
//             int in_row_start = out_y * stride_y - padding_y;
//             int in_col_start = out_x * stride_x - padding_x;
//             int in_row_end = in_row_start + (kernel_h - 1) * dilation_y + 1;
//             int in_col_end = in_col_start + (kernel_w - 1) * dilation_x + 1;
            
//             // Check if fully inside
//             bool fully_inside = 
//                 (in_row_start >= 0) && (in_col_start >= 0) &&
//                 (in_row_end <= in_h) && (in_col_end <= in_w);
                
//             if (fully_inside) {
//                 fully_inside_count++;
//             }
//         }
//     }
    
//     BranchRatio ratio;
//     ratio.fully_inside_ratio = (float)fully_inside_count / total_count * 100.0f;
//     ratio.boundary_ratio = (float)(total_count - fully_inside_count) / total_count * 100.0f;
    
//     return ratio;
// }

// // Calculate and print branch ratio (will be defined after g_log_file)
// void print_branch_ratio(const char* kernel_name, const BranchRatio& ratio);

// // ===== KERNEL IMPLEMENTATIONS =====
// // Copy the necessary structures and kernels from conv2d-dw.cu and conv2d-dw-backup.cu
// struct conv_params {
//     int in_w, in_h;
//     int out_w, out_h;
//     int kernel_w, kernel_h;
//     int stride_x, stride_y;
//     int padding_x, padding_y;
//     int dilation_x, dilation_y;
//     int channels, batches;
// };

// struct kernel_bounds {
//     int y_min, y_max;
//     int x_min, x_max;
// };

// __device__ __forceinline__ kernel_bounds calculate_kernel_bounds(int out_x, int out_y, const conv_params & params) {
//     kernel_bounds bounds;
//     bounds.y_min = max(0, (params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
//     bounds.y_max =
//         min(params.kernel_h,
//             (params.in_h + params.padding_y - out_y * params.stride_y + params.dilation_y - 1) / params.dilation_y);
//     bounds.x_min = max(0, (params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
//     bounds.x_max =
//         min(params.kernel_w,
//             (params.in_w + params.padding_x - out_x * params.stride_x + params.dilation_x - 1) / params.dilation_x);
//     return bounds;
// }

// __device__ __forceinline__ int calculate_input_coord(int out_coord, int kern_coord, int stride, int dilation, int padding) {
//     return out_coord * stride + kern_coord * dilation - padding;
// }

// struct whcn_layout {
//     __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
//         return n * (params.channels * params.in_w * params.in_h) + c * params.in_w * params.in_h + y * params.in_w + x;
//     }

//     __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
//         return c * params.kernel_h * params.kernel_w + ky * params.kernel_w + kx;
//     }

//     __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
//         return n * (params.channels * params.out_w * params.out_h) + c * params.out_w * params.out_h +
//                y * params.out_w + x;
//     }

//     __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
//                                           int & out_x) {
//         out_x = global_idx % params.out_w;
//         out_y = (global_idx / params.out_w) % params.out_h;
//         c     = (global_idx / (params.out_w * params.out_h)) % params.channels;
//         n     = global_idx / (params.out_w * params.out_h * params.channels);
//     }
// };

// struct cwhn_layout {
//     __device__ static int input_index(int n, int c, int y, int x, const conv_params & params) {
//         return n * (params.channels * params.in_w * params.in_h) + (y * params.in_w + x) * params.channels + c;
//     }

//     __device__ static int kernel_index(int c, int ky, int kx, const conv_params & params) {
//         return (ky * params.kernel_w + kx) * params.channels + c;
//     }

//     __device__ static int output_index(int n, int c, int y, int x, const conv_params & params) {
//         return n * (params.channels * params.out_w * params.out_h) + y * (params.out_w * params.channels) +
//                x * params.channels + c;
//     }

//     __device__ static void unpack_indices(int global_idx, const conv_params & params, int & n, int & c, int & out_y,
//                                           int & out_x) {
//         c     = global_idx % params.channels;
//         out_x = (global_idx / params.channels) % params.out_w;
//         out_y = (global_idx / (params.channels * params.out_w)) % params.out_h;
//         n     = global_idx / (params.channels * params.out_w * params.out_h);
//     }
// };

// // ===== CONV2D-DW.CU KERNEL IMPLEMENTATION =====
// template <typename T, typename Layout>
// __global__ void conv2d_dw_kernel(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
//                                  const int in_w, const int in_h, const int out_w, const int out_h,
//                                  const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
//                                  const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
//                                  const int channels, const int batches) {
//     const int global_idx     = blockIdx.x * blockDim.x + threadIdx.x;
//     const int total_elements = batches * channels * out_h * out_w;
//     if (global_idx >= total_elements) return;

//     conv_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                            stride_x, stride_y, padding_x, padding_y,
//                            dilation_x, dilation_y, channels, batches };

//     int n, c, oy, ox;
//     Layout::unpack_indices(global_idx, params, n, c, oy, ox);

//     T acc = 0;

//     const int in_row_start = oy * params.stride_y - params.padding_y;
//     const int in_col_start = ox * params.stride_x - params.padding_x;
//     const int in_row_end   = in_row_start + (params.kernel_h - 1) * params.dilation_y + 1;
//     const int in_col_end   = in_col_start + (params.kernel_w - 1) * params.dilation_x + 1;

//     const bool fully_inside =
//         (in_row_start >= 0) && (in_col_start >= 0) &&
//         (in_row_end   <= params.in_h) && (in_col_end   <= params.in_w);

//     unsigned warp_mask = __activemask();
//     const bool warp_all_inside = __all_sync(warp_mask, fully_inside);

//     if (warp_all_inside) {
//         // ---- FAST PATH: No boundary check, fixed size/fully converged ----
//         #pragma unroll
//         for (int ky = 0; ky < params.kernel_h; ++ky) {
//             const int iy = in_row_start + ky * params.dilation_y;
//             #pragma unroll
//             for (int kx = 0; kx < params.kernel_w; ++kx) {
//                 const int ix = in_col_start + kx * params.dilation_x;
//                 const T ival = input[Layout::input_index(n, c, iy, ix, params)];
//                 const T wval = kernel[Layout::kernel_index(c, ky, kx, params)];
//                 acc += ival * wval;
//             }
//         }
//     } else {
//         // ---- SLOW PATH: Boundary check ----
//         #pragma unroll
//         for (int ky = 0; ky < params.kernel_h; ++ky) {
//             const int iy = in_row_start + ky * params.dilation_y;
//             const bool row_valid = (unsigned)iy < (unsigned)params.in_h;

//             #pragma unroll
//             for (int kx = 0; kx < params.kernel_w; ++kx) {
//                 const int ix = in_col_start + kx * params.dilation_x;
//                 const bool valid = row_valid && ((unsigned)ix < (unsigned)params.in_w);

//                 T ival = 0;
//                 if (valid) {
//                     ival = input[Layout::input_index(n, c, iy, ix, params)];
//                 }
//                 const T wval = kernel[Layout::kernel_index(c, ky, kx, params)];
//                 acc += ival * wval;
//             }
//         }
//     }

//     output[Layout::output_index(n, c, oy, ox, params)] = acc;
// }

// // Tiled version for stride = 1 (better performance)
// template <typename T, typename Layout, int TILE_X, int TILE_Y>
// __global__ void conv2d_dw_kernel_tiled(
//     const T* __restrict__ input,
//     const T* __restrict__ kernel,
//     T* __restrict__ output,
//     int in_w, int in_h,
//     int out_w, int out_h,
//     int kernel_w, int kernel_h,
//     int stride_x, int stride_y,
//     int padding_x, int padding_y,
//     int dilation_x, int dilation_y,
//     int channels, int batches)
// {
//     const int cz = blockIdx.z;
//     const int c  = cz % channels;
//     const int n  = cz / channels;
//     if (n >= batches) return;

//     const int ox0 = blockIdx.x * TILE_X;
//     const int oy0 = blockIdx.y * TILE_Y;

//     const int in_x0 = ox0 * stride_x - padding_x;
//     const int in_y0 = oy0 * stride_y - padding_y;

//     const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
//     const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;

//     extern __shared__ T smem[]; // tile_w * tile_h + kernel_w * kernel_h
//     T* tile = smem;
//     T* kernel_tile = smem + tile_w * tile_h;

//     conv_params params = {
//         in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//         stride_x, stride_y, padding_x, padding_y,
//         dilation_x, dilation_y, channels, batches
//     };

//     // Load input tile to shared memory
//     for (int dy = threadIdx.y; dy < tile_h; dy += blockDim.y) {
//         const int gy = in_y0 + dy;
//         const bool y_ok = (unsigned)gy < (unsigned)in_h;

//         for (int dx = threadIdx.x; dx < tile_w; dx += blockDim.x) {
//             const int gx = in_x0 + dx;
//             T v = 0;
//             if (y_ok && (unsigned)gx < (unsigned)in_w) {
//                 v = input[Layout::input_index(n, c, gy, gx, params)];
//             }
//             tile[dy * tile_w + dx] = v;
//         }
//     }
    
//     // Load kernel to shared memory
//     for (int ky = threadIdx.y; ky < kernel_h; ky += blockDim.y) {
//         for (int kx = threadIdx.x; kx < kernel_w; kx += blockDim.x) {
//             kernel_tile[ky * kernel_w + kx] = kernel[Layout::kernel_index(c, ky, kx, params)];
//         }
//     }
//     __syncthreads();

//     const int ox = ox0 + threadIdx.x;
//     const int oy = oy0 + threadIdx.y;

//     if (ox < out_w && oy < out_h) {
//         T acc = 0;

//         const int sx0 = threadIdx.x * stride_x;
//         const int sy0 = threadIdx.y * stride_y;

//         #pragma unroll
//         for (int ky = 0; ky < kernel_h; ++ky) {
//             const int sy = sy0 + ky * dilation_y;
//             #pragma unroll
//             for (int kx = 0; kx < kernel_w; ++kx) {
//                 const int sx = sx0 + kx * dilation_x;
//                 const T a = tile[sy * tile_w + sx];    
//                 const T w = kernel_tile[ky * kernel_w + kx]; 
//                 acc = fmaf(a, w, acc);
//             }
//         }

//         output[Layout::output_index(n, c, oy, ox, params)] = acc;
//     }
// }

// // ===== CONV2D-DW-BACKUP.CU KERNEL IMPLEMENTATION =====
// template <typename T, typename Layout>
// __global__ void conv2d_dw_kernel_backup(const T * __restrict__ input, const T * __restrict__ kernel, T * __restrict__ output,
//                                         const int in_w, const int in_h, const int out_w, const int out_h,
//                                         const int kernel_w, const int kernel_h, const int stride_x, const int stride_y,
//                                         const int padding_x, const int padding_y, const int dilation_x, const int dilation_y,
//                                         const int channels, const int batches) {
//     const int global_idx     = blockIdx.x * blockDim.x + threadIdx.x;
//     const int total_elements = batches * channels * out_h * out_w;

//     if (global_idx >= total_elements) {
//         return;
//     }

//     conv_params params = { in_w,     in_h,      out_w,     out_h,      kernel_w,   kernel_h, stride_x,
//                            stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches };

//     int batch_idx, channel_idx, out_y_idx, out_x_idx;
//     Layout::unpack_indices(global_idx, params, batch_idx, channel_idx, out_y_idx, out_x_idx);

//     T accumulator = 0;
    
//     kernel_bounds bounds = calculate_kernel_bounds(out_x_idx, out_y_idx, params);

//     for (int kern_y = bounds.y_min; kern_y < bounds.y_max; ++kern_y) {
//         int in_y_idx = calculate_input_coord(out_y_idx, kern_y, params.stride_y, params.dilation_y, params.padding_y);

//         for (int kern_x = bounds.x_min; kern_x < bounds.x_max; ++kern_x) {
//             int in_x_idx = calculate_input_coord(out_x_idx, kern_x, params.stride_x, params.dilation_x, params.padding_x);

//             const T input_val  = input[Layout::input_index(batch_idx, channel_idx, in_y_idx, in_x_idx, params)];
//             const T kernel_val = kernel[Layout::kernel_index(channel_idx, kern_y, kern_x, params)];

//             accumulator += input_val * kernel_val;
//         }
//     }

//     output[Layout::output_index(batch_idx, channel_idx, out_y_idx, out_x_idx, params)] = accumulator;
// }

// // Global log file pointer for benchmark function
// FILE* g_log_file = nullptr;

// // Calculate and print branch ratio
// void print_branch_ratio(const char* kernel_name, const BranchRatio& ratio) {
//     printf("  %s: Fully inside: %.1f%%, Boundary: %.1f%%\n", 
//            kernel_name, ratio.fully_inside_ratio, ratio.boundary_ratio);
    
//     if (g_log_file) {
//         fprintf(g_log_file, "  %s: Fully inside: %.1f%%, Boundary: %.1f%%\n", 
//                 kernel_name, ratio.fully_inside_ratio, ratio.boundary_ratio);
//     }
// }

// // ===== BENCHMARK FUNCTIONS =====
// void benchmark_conv2d_dw(const char* kernel_name, 
//                          const float* input, const float* kernel, float* output,
//                          int in_w, int in_h, int out_w, int out_h,
//                          int kernel_w, int kernel_h, int stride_x, int stride_y,
//                          int padding_x, int padding_y, int dilation_x, int dilation_y,
//                          int channels, int batches, int iterations) {
    
//     printf("Benchmarking %s...\n", kernel_name);
    
//     // Calculate branch ratio
//     BranchRatio ratio = calculate_branch_ratio_analytical(
//         in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//         stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y
//     );
    
//     print_branch_ratio(kernel_name, ratio);
    
//     // Warm-up run
//     if (strcmp(kernel_name, "conv2d_dw_kernel") == 0) {
//         // Choose kernel based on stride (same logic as conv2d-dw.cu)
//         if (stride_x == 1 && stride_y == 1) {
//             // Use tiled kernel for stride = 1 (better performance)
//             const int TILE_X = 32;
//             const int TILE_Y = 8;
            
//             // Calculate tile dimensions
//             const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
//             const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;
//             const size_t shared_mem_size = (tile_w * tile_h + kernel_w * kernel_h) * sizeof(float);
            
//             // Check if shared memory size is reasonable (fallback to regular kernel if too large)
//             const size_t max_shared_mem = 32 * 1024; // 32KB limit
            
//             if (shared_mem_size <= max_shared_mem) {
//                 // 2D block configuration for tiled kernel
//                 dim3 blocks_2d((out_w + TILE_X - 1) / TILE_X, 
//                                 (out_h + TILE_Y - 1) / TILE_Y, 
//                                 batches * channels);
//                 dim3 threads_2d(TILE_X, TILE_Y);
                
//                 conv2d_dw_kernel_tiled<float, whcn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size>>>(
//                     input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                     stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//             } else {
//                 // Fallback to regular kernel if shared memory is too large
//                 const int total = batches * channels * out_h * out_w;
//                 const int blocks = (total + 256 - 1) / 256;
                
//                 conv2d_dw_kernel<float, whcn_layout><<<blocks, 256>>>(
//                     input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                     stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//             }
//         } else {
//             // Use regular kernel for stride > 1
//             const int total = batches * channels * out_h * out_w;
//             const int blocks = (total + 256 - 1) / 256;
            
//             conv2d_dw_kernel<float, whcn_layout><<<blocks, 256>>>(
//                 input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                 stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//         }

//     } else if (strcmp(kernel_name, "conv2d_dw_kernel_backup") == 0) {
//         const int total = batches * channels * out_h * out_w;
//         const int blocks = (total + 256 - 1) / 256;
        
//         conv2d_dw_kernel_backup<float, whcn_layout><<<blocks, 256>>>(
//             input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//             stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//     }
    
//     BENCHMARK_CUDA_CHECK(cudaDeviceSynchronize());
    
//     // Timed runs
//     std::vector<float> durations;
//     durations.reserve(iterations);
    
//     for (int i = 0; i < iterations; ++i) {
//         cudaEvent_t start, stop;
//         BENCHMARK_CUDA_CHECK(cudaEventCreate(&start));
//         BENCHMARK_CUDA_CHECK(cudaEventCreate(&stop));
        
//         BENCHMARK_CUDA_CHECK(cudaEventRecord(start));
        
//         if (strcmp(kernel_name, "conv2d_dw_kernel") == 0) {
//             // Choose kernel based on stride (same logic as conv2d-dw.cu)
//             if (stride_x == 1 && stride_y == 1) {
//                 // Use tiled kernel for stride = 1 (better performance)
//                 const int TILE_X = 32;
//                 const int TILE_Y = 8;
                
//                 // Calculate tile dimensions
//                 const int tile_w = (TILE_X - 1) * stride_x + (kernel_w - 1) * dilation_x + 1;
//                 const int tile_h = (TILE_Y - 1) * stride_y + (kernel_h - 1) * dilation_y + 1;
//                 const size_t shared_mem_size = (tile_w * tile_h + kernel_w * kernel_h) * sizeof(float);
                
//                 // Check if shared memory size is reasonable (fallback to regular kernel if too large)
//                 const size_t max_shared_mem = 32 * 1024; // 32KB limit
                
//                 if (shared_mem_size <= max_shared_mem) {
//                     // 2D block configuration for tiled kernel
//                     dim3 blocks_2d((out_w + TILE_X - 1) / TILE_X, 
//                                     (out_h + TILE_Y - 1) / TILE_Y, 
//                                     batches * channels);
//                     dim3 threads_2d(TILE_X, TILE_Y);
                    
//                     conv2d_dw_kernel_tiled<float, whcn_layout, TILE_X, TILE_Y><<<blocks_2d, threads_2d, shared_mem_size>>>(
//                         input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                         stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//                 } else {
//                     // Fallback to regular kernel if shared memory is too large
//                     const int total = batches * channels * out_h * out_w;
//                     const int blocks = (total + 256 - 1) / 256;
                    
//                     conv2d_dw_kernel<float, whcn_layout><<<blocks, 256>>>(
//                         input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                         stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//                 }
//             } else {
//                 // Use regular kernel for stride > 1
//                 const int total = batches * channels * out_h * out_w;
//                 const int blocks = (total + 256 - 1) / 256;
                
//                 conv2d_dw_kernel<float, whcn_layout><<<blocks, 256>>>(
//                     input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                     stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//             }

//         } else if (strcmp(kernel_name, "conv2d_dw_kernel_backup") == 0) {
//             const int total = batches * channels * out_h * out_w;
//             const int blocks = (total + 256 - 1) / 256;
            
//             conv2d_dw_kernel_backup<float, whcn_layout><<<blocks, 256>>>(
//                 input, kernel, output, in_w, in_h, out_w, out_h, kernel_w, kernel_h,
//                 stride_x, stride_y, padding_x, padding_y, dilation_x, dilation_y, channels, batches);
//         }
        
//         BENCHMARK_CUDA_CHECK(cudaEventRecord(stop));
//         BENCHMARK_CUDA_CHECK(cudaEventSynchronize(stop));
        
//         float duration;
//         BENCHMARK_CUDA_CHECK(cudaEventElapsedTime(&duration, start, stop));
//         durations.push_back(duration);
        
//         BENCHMARK_CUDA_CHECK(cudaEventDestroy(start));
//         BENCHMARK_CUDA_CHECK(cudaEventDestroy(stop));
//     }
    
//     // Calculate statistics
//     float total_time = 0.0f;
//     float min_time = durations[0];
//     float max_time = durations[0];
    
//     for (float d : durations) {
//         total_time += d;
//         if (d < min_time) min_time = d;
//         if (d > max_time) max_time = d;
//     }
    
//     float avg_time = total_time / iterations;
    
//     printf("  %s: Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.4f ms\n", 
//            kernel_name, avg_time, min_time, max_time, total_time);
    
//     if (g_log_file) {
//         fprintf(g_log_file, "  %s: Avg: %.4f ms, Min: %.4f ms, Max: %.4f ms, Total: %.4f ms\n", 
//                 kernel_name, avg_time, min_time, max_time, total_time);
//     }
// }

// // ===== MAIN FUNCTION =====
// int main(int argc, char* argv[]) {
//     if (argc != 1 && argc != 7) {
//         printf("Usage: %s [input_size kernel_size stride padding dilation iterations]\n", argv[0]);
//         printf("  If no arguments provided, runs all predefined test cases\n");
//         printf("  If 6 arguments provided, runs custom configuration\n");
//         printf("  Example: %s 64 3 1 1 1 100\n", argv[0]);
//         return 1;
//     }
    
//     // Set default iterations
//     int default_iterations = 100;
    
//     if (argc == 7) {
//         // Custom configuration mode
//         int input_size = atoi(argv[1]);
//         int kernel_size = atoi(argv[2]);
//         int stride = atoi(argv[3]);
//         int padding = atoi(argv[4]);
//         int dilation = atoi(argv[5]);
//         int iterations = atoi(argv[6]);
        
//         printf("Custom Configuration:\n");
//         printf("  Input size: %dx%d\n", input_size, input_size);
//         printf("  Kernel size: %dx%d\n", kernel_size, kernel_size);
//         printf("  Stride: %dx%d\n", stride, stride);
//         printf("  Padding: %dx%d\n", padding, padding);
//         printf("  Dilation: %dx%d\n", dilation, dilation);
//         printf("  Iterations: %d\n", iterations);
//         printf("\n");
        
//         // Calculate output dimensions
//         int out_w = (input_size + 2 * padding - (kernel_size - 1) * dilation - 1) / stride + 1;
//         int out_h = (input_size + 2 * padding - (kernel_size - 1) * dilation - 1) / stride + 1;
        
//         if (out_w <= 0 || out_h <= 0) {
//             printf("Error: Invalid configuration results in non-positive output dimensions\n");
//             return 1;
//         }
        
//         // Allocate memory
//         size_t input_size_bytes = input_size * input_size * 3 * 1 * sizeof(float);
//         size_t kernel_size_bytes = kernel_size * kernel_size * 3 * sizeof(float);
//         size_t output_size_bytes = out_w * out_h * 3 * 1 * sizeof(float);
        
//         float *input, *kernel, *output;
//         BENCHMARK_CUDA_CHECK(cudaMalloc(&input, input_size_bytes));
//         BENCHMARK_CUDA_CHECK(cudaMalloc(&kernel, kernel_size_bytes));
//         BENCHMARK_CUDA_CHECK(cudaMalloc(&output, output_size_bytes));
        
//         // Initialize data
//         BENCHMARK_CUDA_CHECK(cudaMemset(input, 0, input_size_bytes));
//         BENCHMARK_CUDA_CHECK(cudaMemset(kernel, 0, kernel_size_bytes));
//         BENCHMARK_CUDA_CHECK(cudaMemset(output, 0, output_size_bytes));
        
//         // Run benchmarks
//         nvtxRangeId_t bench_all = nvtx_push("bench_all");
//         benchmark_conv2d_dw("conv2d_dw_kernel", input, kernel, output,
//                            input_size, input_size, out_w, out_h,
//                            kernel_size, kernel_size, stride, stride,
//                            padding, padding, dilation, dilation,
//                            3, 1, iterations);
//         nvtx_pop(bench_all);
        
//         nvtxRangeId_t bench_backup = nvtx_push("bench_backup");
//         benchmark_conv2d_dw("conv2d_dw_kernel_backup", input, kernel, output,
//                            input_size, input_size, out_w, out_h,
//                            kernel_size, kernel_size, stride, stride,
//                            padding, padding, dilation, dilation,
//                            3, 1, iterations);
//         nvtx_pop(bench_backup);
        
//         // Cleanup
//         BENCHMARK_CUDA_CHECK(cudaFree(input));
//         BENCHMARK_CUDA_CHECK(cudaFree(kernel));
//         BENCHMARK_CUDA_CHECK(cudaFree(output));
        
//     } else {
//         // Full benchmark mode
//         printf("Running full benchmark suite...\n\n");
        
//         // Test cases
//         struct TestCase {
//             int input_size, kernel_size, stride, padding, dilation;
//         };
        
//         TestCase test_cases[] = {
//             {32, 3, 1, 1, 1},
//             {64, 3, 1, 1, 1},
//             {128, 3, 1, 1, 1},
//             {256, 3, 1, 1, 1},
//             {32, 5, 1, 2, 1},
//             {64, 5, 1, 2, 1},
//             {128, 5, 1, 2, 1},
//             {256, 5, 1, 2, 1},
//             {32, 7, 1, 3, 1},
//             {64, 7, 1, 3, 1},
//             {128, 7, 1, 3, 1},
//             {256, 7, 1, 3, 1},
//             {32, 3, 2, 1, 1},
//             {64, 3, 2, 1, 1},
//             {128, 3, 2, 1, 1},
//             {256, 3, 2, 1, 1},
//             {32, 5, 2, 2, 1},
//             {64, 5, 2, 2, 1},
//             {128, 5, 2, 2, 1},
//             {256, 5, 2, 2, 1}
//         };
        
//         for (const auto& test : test_cases) {
//             int out_w = (test.input_size + 2 * test.padding - (test.kernel_size - 1) * test.dilation - 1) / test.stride + 1;
//             int out_h = (test.input_size + 2 * test.padding - (test.kernel_size - 1) * test.dilation - 1) / test.stride + 1;
            
//             printf("Test case: %dx%d input, %dx%d kernel, stride=%d, padding=%d, dilation=%d\n",
//                    test.input_size, test.input_size, test.kernel_size, test.kernel_size,
//                    test.stride, test.padding, test.dilation);
//             printf("Output dimensions: %dx%d\n", out_w, out_h);
            
//             // Allocate memory
//             size_t input_size_bytes = test.input_size * test.input_size * 3 * 1 * sizeof(float);
//             size_t kernel_size_bytes = test.kernel_size * test.kernel_size * 3 * sizeof(float);
//             size_t output_size_bytes = out_w * out_h * 3 * 1 * sizeof(float);
            
//             float *input, *kernel, *output;
//             BENCHMARK_CUDA_CHECK(cudaMalloc(&input, input_size_bytes));
//             BENCHMARK_CUDA_CHECK(cudaMalloc(&kernel, kernel_size_bytes));
//             BENCHMARK_CUDA_CHECK(cudaMalloc(&output, output_size_bytes));
            
//             // Initialize data
//             BENCHMARK_CUDA_CHECK(cudaMemset(input, 0, input_size_bytes));
//             BENCHMARK_CUDA_CHECK(cudaMemset(kernel, 0, kernel_size_bytes));
//             BENCHMARK_CUDA_CHECK(cudaMemset(output, 0, output_size_bytes));
            
//             // Run benchmarks
//             nvtxRangeId_t bench_all = nvtx_push("bench_all");
//             benchmark_conv2d_dw("conv2d_dw_kernel", input, kernel, output,
//                                test.input_size, test.input_size, out_w, out_h,
//                                test.kernel_size, test.kernel_size, test.stride, test.stride,
//                                test.padding, test.padding, test.dilation, test.dilation,
//                                3, 1, default_iterations);
//             nvtx_pop(bench_all);
            

            
//             nvtxRangeId_t bench_backup = nvtx_push("bench_backup");
//             benchmark_conv2d_dw("conv2d_dw_kernel_backup", input, kernel, output,
//                                test.input_size, test.input_size, out_w, out_h,
//                                test.kernel_size, test.kernel_size, test.stride, test.stride,
//                                test.padding, test.padding, test.dilation, test.dilation,
//                                3, 1, default_iterations);
//             nvtx_pop(bench_backup);
            
//             // Cleanup
//             BENCHMARK_CUDA_CHECK(cudaFree(input));
//             BENCHMARK_CUDA_CHECK(cudaFree(kernel));
//             BENCHMARK_CUDA_CHECK(cudaFree(output));
            
//             printf("\n");
//         }
//     }
    
//     printf("Benchmark completed successfully!\n");
//     return 0;
// } 