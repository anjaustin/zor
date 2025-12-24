"""
TriX Native CUDA Kernels

Raw CUDA C code for forward pass, backward pass, and optimization.
No PyTorch. No frameworks. Pure CUDA.
"""

NATIVE_KERNELS = r'''
extern "C" {

// =============================================================================
// SHARED UTILITIES
// =============================================================================

__device__ __forceinline__ float cubic_bspline(float t) {
    t = fabsf(t);
    if (t < 1.0f) return 0.666667f - t*t + 0.5f * t*t*t;
    if (t < 2.0f) { float tmp = 2.0f - t; return 0.166667f * tmp * tmp * tmp; }
    return 0.0f;
}

__device__ __forceinline__ float cubic_bspline_deriv(float t) {
    // Derivative of cubic B-spline w.r.t. t
    float sign = (t < 0) ? -1.0f : 1.0f;
    t = fabsf(t);
    if (t < 1.0f) return sign * (-2.0f * t + 1.5f * t * t);
    if (t < 2.0f) { float tmp = 2.0f - t; return sign * (-0.5f * tmp * tmp); }
    return 0.0f;
}

__device__ __forceinline__ int unpack_ternary(unsigned int packed, int idx) {
    int shift = (idx % 16) * 2;
    int bits = (packed >> shift) & 0x3;
    return bits - 1;
}

// =============================================================================
// FORWARD KERNEL (with saved intermediates for backward)
// =============================================================================

#define BLOCK_SIZE 256
#define MAX_TILES 32
#define MAX_D_MODEL 128

__global__ void forward_with_cache(
    const float* __restrict__ x,
    const unsigned int* __restrict__ signatures,
    const signed char* __restrict__ directions,
    const float* __restrict__ spline_coeffs,
    const float* __restrict__ positions,
    float* __restrict__ output,
    int* __restrict__ tile_indices,      // Save winning tile per token
    float* __restrict__ spline_scales,   // Save spline output per token
    float* __restrict__ spatial_scores,  // Save spatial scores for gradient
    const float dir_scale,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size,
    float position_spread,
    float output_scale
) {
    __shared__ unsigned int s_signatures[MAX_TILES][MAX_D_MODEL / 16];
    __shared__ signed char s_directions[MAX_TILES][MAX_D_MODEL];

    int tid = threadIdx.x;
    int global_tid = blockIdx.x * BLOCK_SIZE + tid;

    // Cooperative load
    int sig_words = num_tiles * (d_model / 16);
    for (int i = tid; i < sig_words; i += BLOCK_SIZE) {
        int t = i / (d_model / 16);
        int w = i % (d_model / 16);
        s_signatures[t][w] = signatures[i];
    }

    int dir_bytes = num_tiles * d_model;
    for (int i = tid; i < dir_bytes; i += BLOCK_SIZE) {
        s_directions[i / d_model][i % d_model] = directions[i];
    }
    __syncthreads();

    if (global_tid >= batch_size) return;

    // Load input
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        x_local[d] = x[global_tid * d_model + d];
    }
    float pos = positions[global_tid];

    // Find best tile
    float best_score = -1e30f;
    int best_tile = 0;

    for (int t = 0; t < num_tiles; t++) {
        float content_score = 0.0f;

        for (int w = 0; w < d_model / 16; w++) {
            unsigned int packed = s_signatures[t][w];
            #pragma unroll 16
            for (int i = 0; i < 16; i++) {
                int sign = unpack_ternary(packed, i);
                content_score += x_local[w * 16 + i] * (float)sign;
            }
        }

        float tile_center = (float)t / (float)num_tiles * 64.0f;
        float spatial = cubic_bspline((pos - tile_center) / position_spread);
        float combined = content_score * spatial;

        // Save spatial score for this tile (for gradient)
        if (t < MAX_TILES) {
            spatial_scores[global_tid * num_tiles + t] = spatial;
        }

        if (combined > best_score) {
            best_score = combined;
            best_tile = t;
        }
    }

    // Save winning tile
    tile_indices[global_tid] = best_tile;

    // Spline lookup
    float a = tanhf(x_local[0] * 0.1f);
    float b = tanhf(x_local[1] * 0.1f);

    int idx_a = min(max((int)(((a + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);
    int idx_b = min(max((int)(((b + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);

    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;
    float base = spline_coeffs[coeff_idx];
    float slope_a = spline_coeffs[coeff_idx + 1];
    float slope_b = spline_coeffs[coeff_idx + 2];

    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;
    float scale = (base + slope_a * local_a + slope_b * local_b) * output_scale;

    // Save spline scale
    spline_scales[global_tid] = scale;

    // Output with residual
    for (int d = 0; d < d_model; d++) {
        float dir = (float)s_directions[best_tile][d] * dir_scale;
        output[global_tid * d_model + d] = x_local[d] + scale * dir;
    }
}

// =============================================================================
// BACKWARD KERNEL
// =============================================================================

__global__ void backward_kernel(
    const float* __restrict__ d_output,     // [batch, d_model] gradient from loss
    const float* __restrict__ x,            // [batch, d_model] original input
    const int* __restrict__ tile_indices,   // [batch] winning tile per token
    const float* __restrict__ spline_scales,// [batch] scale per token
    const signed char* __restrict__ directions, // [num_tiles, d_model]
    float* __restrict__ d_directions,       // [num_tiles, d_model] gradient accumulator
    float* __restrict__ d_spline_coeffs,    // [num_tiles, grid*grid*3] gradient accumulator
    float* __restrict__ d_input,            // [batch, d_model] gradient to input
    const float dir_scale,
    const float output_scale,
    int batch_size,
    int d_model,
    int num_tiles,
    int grid_size
) {
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid >= batch_size) return;

    int best_tile = tile_indices[global_tid];
    float scale = spline_scales[global_tid];

    // Load gradient and input
    float d_out_local[MAX_D_MODEL];
    float x_local[MAX_D_MODEL];
    for (int d = 0; d < d_model; d++) {
        d_out_local[d] = d_output[global_tid * d_model + d];
        x_local[d] = x[global_tid * d_model + d];
    }

    // Gradient through residual connection
    // output = x + scale * direction
    // d_input = d_output (residual passes through)
    for (int d = 0; d < d_model; d++) {
        d_input[global_tid * d_model + d] = d_out_local[d];
    }

    // Gradient w.r.t. directions (accumulated atomically)
    // d_output/d_direction = scale * dir_scale
    for (int d = 0; d < d_model; d++) {
        float grad = d_out_local[d] * scale / dir_scale;  // Chain rule
        atomicAdd(&d_directions[best_tile * d_model + d], grad);
    }

    // Gradient w.r.t. spline coefficients
    // output = x + (base + slope_a * local_a + slope_b * local_b) * output_scale * direction
    // We need d_output/d_base, d_output/d_slope_a, d_output/d_slope_b

    float a = tanhf(x_local[0] * 0.1f);
    float b = tanhf(x_local[1] * 0.1f);

    int idx_a = min(max((int)(((a + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);
    int idx_b = min(max((int)(((b + 1.0f) / 2.0f) * grid_size), 0), grid_size - 1);

    float cell_size = 2.0f / (float)grid_size;
    float local_a = ((a + 1.0f) - idx_a * cell_size) / cell_size;
    float local_b = ((b + 1.0f) - idx_b * cell_size) / cell_size;

    // Sum of (d_output * direction) gives us d_loss/d_scale
    float d_scale = 0.0f;
    for (int d = 0; d < d_model; d++) {
        float dir = (float)directions[best_tile * d_model + d] * dir_scale;
        d_scale += d_out_local[d] * dir;
    }

    // Now chain through spline
    // scale = (base + slope_a * local_a + slope_b * local_b) * output_scale
    // d_scale/d_base = output_scale
    // d_scale/d_slope_a = output_scale * local_a
    // d_scale/d_slope_b = output_scale * local_b

    int coeff_idx = best_tile * grid_size * grid_size * 3 + idx_a * grid_size * 3 + idx_b * 3;

    atomicAdd(&d_spline_coeffs[coeff_idx + 0], d_scale * output_scale);
    atomicAdd(&d_spline_coeffs[coeff_idx + 1], d_scale * output_scale * local_a);
    atomicAdd(&d_spline_coeffs[coeff_idx + 2], d_scale * output_scale * local_b);
}

// =============================================================================
// WEIGHT UPDATE KERNEL (Adam optimizer fused)
// =============================================================================

__global__ void adam_update(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float beta1_t,  // beta1^t for bias correction
    float beta2_t,  // beta2^t for bias correction
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx];

    // Update biased moments
    float m_new = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_new = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = m_new;
    v[idx] = v_new;

    // Bias correction
    float m_hat = m_new / (1.0f - beta1_t);
    float v_hat = v_new / (1.0f - beta2_t);

    // Update parameter
    param[idx] -= lr * m_hat / (sqrtf(v_hat) + eps);
}

// =============================================================================
// GRADIENT ZEROING
// =============================================================================

__global__ void zero_gradients(float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) grad[idx] = 0.0f;
}

}  // extern "C"
'''
