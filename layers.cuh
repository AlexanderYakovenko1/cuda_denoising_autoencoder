#ifndef LAYERS_CUH
#define LAYERS_CUH

#include <cstdio>

/// \brief Wrapper macro for CUDA error checking
///
/// Usage: CHECK_CUDA_ERRS( cudaMalloc(...) );
#define CHECK_CUDA_ERRS(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr,"CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__
void Conv2D_3x3(const float* in, float* out, const float* kernel_W, const float* kernel_b, int in_channels, int out_channels, int in_height, int in_width);

__global__
void TranposedConv2D_3x3_2(const float* in, float* out, float* buf, const float* kernel_W, const float* kernel_b, int in_channels, int out_channels, int in_height, int in_width);

__global__
void MaxPool2D(const float* in, float* out, int in_channels, int in_height, int in_width, int scale);

__global__
void ReLU(const float* in, float* out);

__global__
void Sigmoid(const float* in, float* out);

#endif //LAYERS_CUH
