#include "layers.cuh"


__global__
void ApplyConv2D__basic(const float* src, float* dst, int width, int height, int channels, const float* kernel, int kernel_width, int kernel_height) {
    int kernel_radius_width = kernel_width / 2;
    int kernel_radius_height = kernel_height / 2;

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < height && j < width) {
        float conv_val = 0.;
        size_t idx, k_idx;
        for (int k_i = -kernel_radius_height; k_i <= kernel_radius_height; ++k_i) {
            for (int k_j = -kernel_radius_width; k_j <= kernel_radius_width; ++k_j) {
                if (i + k_i >= 0 && i + k_i < height && j + k_j >= 0 && j + k_j < width) {
                    idx = (i + k_i) * width + j + k_j;
                    k_idx = (k_i + kernel_radius_height) * kernel_width + k_j + kernel_radius_width;

                    conv_val += src[idx] * kernel[k_idx];
                }
            }
        }

        idx = i * width + j;
        dst[idx] = conv_val;
    }
}

constexpr int CONV_3x3_RADIUS = 1;
constexpr int CONV_3x3_DIM = 3;

/// does not change H, W dims
/// zero-padding at edges
__global__
void Conv2D_3x3(const float* in, float* out, const float* kernel_W, const float* kernel_b, int in_channels, int out_channels, int in_height, int in_width) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < in_height && j < in_width) {
        int in_channel_size = in_height * in_width;
        int in_ch_size = CONV_3x3_DIM * CONV_3x3_DIM;
        int out_ch_size = in_channels * in_ch_size;

        for (int out_ch = 0; out_ch < out_channels; ++ out_ch) {

            float conv_val = 0.;
            size_t idx, k_idx;
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {

                for (int k_i = -CONV_3x3_RADIUS; k_i <= CONV_3x3_RADIUS; ++k_i) { // kernel height
                    for (int k_j = -CONV_3x3_RADIUS; k_j <= CONV_3x3_RADIUS; ++k_j) { // kernel width
                        if (i + k_i >= 0 && i + k_i < in_height && j + k_j >= 0 && j + k_j < in_width) {
                            idx = in_ch * in_channel_size + (i + k_i) * in_width + j + k_j;
                            k_idx = out_ch * out_ch_size + in_ch * in_ch_size + (k_i + CONV_3x3_RADIUS) * CONV_3x3_DIM + k_j + CONV_3x3_RADIUS;

                            conv_val += in[idx] * kernel_W[k_idx];
                        }
                    }
                }

            }

            idx = out_ch * in_channel_size + i * in_width + j;
            out[idx] = conv_val + kernel_b[out_ch];
        }
    }
}

/// called for output image size
/// i.e. numBlocks.x = out_width and NOT in_width
__global__
void TranposedConv2D_3x3_2(const float* in, float* out, float* buf, const float* kernel_W, const float* kernel_b, int in_channels, int out_channels, int in_height, int in_width) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = 2 * in_height;
    int out_width = 2 * in_width;

    if (i < out_height && j < out_width) { // load spaced out input into buffer to convolve over
        for (int out_ch = 0; out_ch < out_channels; ++out_ch) {
            if ((i & 1) == 0 && (j & 1) == 0) {  // same as i % 2 == 0 && j % 2 == 0
                buf[out_ch * out_height * out_width + i * out_width + j] = in[out_ch * in_width * in_height +
                                                                              (i >> 1) * in_width + (j >> 1)];
            } else {
                buf[out_ch * out_height * out_width + i * out_width + j] = 0.f;
            }
        }
    }
    __syncthreads();

    if (i < out_height && j < out_width) {
        int out_channel_size = out_height * out_width;
        int out_ch_size = CONV_3x3_DIM * CONV_3x3_DIM;
        int in_ch_size = in_channels * out_ch_size;

        for (int out_ch = 0; out_ch < out_channels; ++ out_ch) {

            float conv_val = 0.;
            size_t idx, k_idx;
            for (int in_ch = 0; in_ch < in_channels; ++in_ch) {

                for (int k_i = -CONV_3x3_RADIUS; k_i <= CONV_3x3_RADIUS; ++k_i) { // kernel height
                    for (int k_j = -CONV_3x3_RADIUS; k_j <= CONV_3x3_RADIUS; ++k_j) { // kernel width
                        if (i + k_i >= 0 && i + k_i < out_height && j + k_j >= 0 && j + k_j < out_width) {
                            idx = in_ch * out_channel_size + (i + k_i) * out_width + j + k_j;
                            // reversed order in comparison to regular conv
                            k_idx = out_ch * out_ch_size + in_ch * in_ch_size + (CONV_3x3_RADIUS - k_i) * CONV_3x3_DIM + CONV_3x3_RADIUS - k_j;

                            conv_val += buf[idx] * kernel_W[k_idx];
                        }
                    }
                }

            }

            idx = out_ch * out_channel_size + i * out_width + j;
            out[idx] = conv_val + kernel_b[out_ch];
        }
    }
}

// todo: try pitched arrays

/// called for output image size
/// i.e. numBlocks.x = out_width and NOT in_width
__global__
void MaxPool2D(const float* in, float* out, int in_channels, int in_height, int in_width, int scale) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    int out_height = in_height / scale;
    int out_width = in_width / scale;

    if (i < out_height && j < out_width) {
        int in_channel_size = in_height * in_width;
        int out_channel_size = out_height * out_width;

        for (int in_ch = 0; in_ch < in_channels; ++in_ch) {

            float max_val = 0.;
            size_t in_idx, out_idx;

            for (int k_i = 0; k_i < scale; ++k_i) {
                for (int k_j = 0; k_j < scale; ++k_j) {

                    int in_i = i * scale + k_i;
                    int in_j = j * scale + k_j;
                    if (in_i < in_height && in_j < in_width) {
                        in_idx = in_ch * in_channel_size + in_i * in_width + in_j;
                        max_val = max(max_val, in[in_idx]);
                    }

                }
            }

            out_idx = in_ch * out_channel_size + i * out_width + j;
            out[out_idx] = max_val;
        }
    }
}

__global__
void ReLU(const float* in, float* out, int in_channels, int in_height, int in_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_channels * in_height * in_width) {
        out[idx] = max(in[idx], 0.f);
    }
}

__global__
void Sigmoid(const float* in, float* out, int in_channels, int in_height, int in_width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < in_channels * in_height * in_width) {
        out[idx] = 1.f / (1.f + exp(-in[idx]));
    }
}
