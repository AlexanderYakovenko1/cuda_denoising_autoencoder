#include <iostream>
#include <vector>

#include "include/cnpy.h"
#include "utils.h"
#include "layers.cuh"

constexpr int BLOCK_SIZE = 32;
constexpr int MAX_THREADS = 1024;

std::pair<int64_t, int64_t> DenoiseMNIST(const float* in, float* out, int height, int width, const std::string& weights_dir="../model_weights") {
    std::chrono::steady_clock::time_point all_begin = std::chrono::steady_clock::now();

    cnpy::NpyArray conv1_weight = cnpy::npy_load(weights_dir + "/conv1.weight.npy");
    cnpy::NpyArray conv1_bias = cnpy::npy_load(weights_dir + "/conv1.bias.npy");
    cnpy::NpyArray conv2_weight = cnpy::npy_load(weights_dir + "/conv2.weight.npy");
    cnpy::NpyArray conv2_bias = cnpy::npy_load(weights_dir + "/conv2.bias.npy");

    cnpy::NpyArray upconv1_weight = cnpy::npy_load(weights_dir + "/upconv1.weight.npy");
    cnpy::NpyArray upconv1_bias = cnpy::npy_load(weights_dir + "/upconv1.bias.npy");
    cnpy::NpyArray upconv2_weight = cnpy::npy_load(weights_dir + "/upconv2.weight.npy");
    cnpy::NpyArray upconv2_bias = cnpy::npy_load(weights_dir + "/upconv2.bias.npy");
    cnpy::NpyArray conv3_weight = cnpy::npy_load(weights_dir + "/conv3.weight.npy");
    cnpy::NpyArray conv3_bias = cnpy::npy_load(weights_dir + "/conv3.bias.npy");

    float* cuda_conv1_weight = nullptr;
    float* cuda_conv1_bias = nullptr;
    float* cuda_conv2_weight = nullptr;
    float* cuda_conv2_bias = nullptr;

    float* cuda_upconv1_weight = nullptr;
    float* cuda_upconv1_bias = nullptr;
    float* cuda_upconv2_weight = nullptr;
    float* cuda_upconv2_bias = nullptr;
    float* cuda_conv3_weight = nullptr;
    float* cuda_conv3_bias = nullptr;

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv1_weight, conv1_weight.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv1_weight, conv1_weight.data<float>(), conv1_weight.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv1_bias, conv1_bias.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv1_bias, conv1_bias.data<float>(), conv1_bias.num_vals * sizeof(float), cudaMemcpyHostToDevice) );

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv2_weight, conv2_weight.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv2_weight, conv2_weight.data<float>(), conv2_weight.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv2_bias, conv2_bias.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv2_bias, conv2_bias.data<float>(), conv2_bias.num_vals * sizeof(float), cudaMemcpyHostToDevice) );

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_upconv1_weight, upconv1_weight.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_upconv1_weight, upconv1_weight.data<float>(), upconv1_weight.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_upconv1_bias, upconv1_bias.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_upconv1_bias, upconv1_bias.data<float>(), upconv1_bias.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_upconv2_weight, upconv2_weight.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_upconv2_weight, upconv2_weight.data<float>(), upconv2_weight.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_upconv2_bias, upconv2_bias.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_upconv2_bias, upconv2_bias.data<float>(), upconv2_bias.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv3_weight, conv3_weight.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv3_weight, conv3_weight.data<float>(), conv3_weight.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_conv3_bias, conv3_bias.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_conv3_bias, conv3_bias.data<float>(), conv3_bias.num_vals * sizeof(float), cudaMemcpyHostToDevice) );

    int half_width = width >> 1;
    int half_height = height >> 1;
    int quarter_width = width >> 2;
    int quarter_height = height >> 2;
    int full_size = width * height;
    int half_size = half_width * half_height;
    int quarter_size = quarter_width * quarter_height;

    float* cuda_fullres1 = nullptr;
    float* cuda_fullres32_1 = nullptr;
    float* cuda_fullres32_2 = nullptr;
    float* cuda_halfres32_1 = nullptr;
    float* cuda_halfres32_2 = nullptr;
    float* cuda_quarterres32 = nullptr;

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres1, full_size * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_fullres1, in, full_size * sizeof(float), cudaMemcpyHostToDevice) );

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres32_1, 32 * full_size * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres32_2, 32 * full_size * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_halfres32_1, 32 * half_size * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_halfres32_2, 32 * half_size * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_quarterres32, 32 * quarter_size * sizeof(float)) );

//
    // todo: remore, this is for debug only
//    float* cpu_out = static_cast<float *>(malloc(32 * full_size * sizeof(float)));
//

    dim3 numBlocksFull(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    dim3 numBlocksHalf(half_width / BLOCK_SIZE + 1, half_height / BLOCK_SIZE + 1);
    dim3 numBlocksQuarter(quarter_width / BLOCK_SIZE + 1, quarter_height / BLOCK_SIZE + 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

    Conv2D_3x3<<<numBlocksFull, threadsPerBlock>>>(
            cuda_fullres1, cuda_fullres32_1, cuda_conv1_weight, cuda_conv1_bias, 1, 32, height, width);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error1: %s\n", cudaGetErrorString(error)); exit(-1);
    }

//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres32_1, 32 * full_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../0_conv1.npy", cpu_out, {32, 256, 256});
//

    ReLU<<<32 * full_size / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres32_1, cuda_fullres32_1, 32, height, width);

    MaxPool2D<<<numBlocksHalf, threadsPerBlock>>>(cuda_fullres32_1, cuda_halfres32_1, 32, height, width, 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error2: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_1, 32 * half_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../1_maxpool1.npy", cpu_out, {32, 128, 128});
//

    Conv2D_3x3<<<numBlocksHalf, threadsPerBlock>>>(
            cuda_halfres32_1, cuda_halfres32_2, cuda_conv2_weight, cuda_conv2_bias, 32, 32, half_height, half_width);
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_2, 32 * half_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../2_conv2.npy", cpu_out, {32, 128, 128});
//
    ReLU<<<32 * half_size / MAX_THREADS + 1, MAX_THREADS>>>(cuda_halfres32_2, cuda_halfres32_2, 32, half_height, half_width);

    MaxPool2D<<<numBlocksQuarter, threadsPerBlock>>>(cuda_halfres32_2, cuda_quarterres32, 32, half_height, half_width, 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error3: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_quarterres32, 32 * quarter_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../3_maxpool2.npy", cpu_out, {32, 64, 64});
//

    TranposedConv2D_3x3_2<<<numBlocksHalf, threadsPerBlock>>>(
            cuda_quarterres32, cuda_halfres32_1, cuda_halfres32_2, cuda_upconv1_weight, cuda_upconv1_bias, 32, 32, quarter_height, quarter_width);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error4: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_1, 32 * half_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../4_upconv1.npy", cpu_out, {32, 128, 128});
//
    ReLU<<<32 * half_size / MAX_THREADS + 1, MAX_THREADS>>>(cuda_halfres32_1, cuda_halfres32_1, 32, half_height, half_width);

    TranposedConv2D_3x3_2<<<numBlocksFull, threadsPerBlock>>>(
            cuda_halfres32_1, cuda_fullres32_1, cuda_fullres32_2, cuda_upconv2_weight, cuda_upconv2_bias, 32, 32, half_height, half_width);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error5: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres32_1, 32 * full_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../5_upconv2.npy", cpu_out, {32, 256, 256});
//
    ReLU<<<32 * full_size / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres32_1, cuda_fullres32_1, 32, height, width);

    Conv2D_3x3<<<numBlocksFull, threadsPerBlock>>>(
            cuda_fullres32_1, cuda_fullres1, cuda_conv3_weight, cuda_conv3_bias, 32, 1, height, width);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error6: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres1, full_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../6_conv3.npy", cpu_out, {1, 256, 256});
//
    Sigmoid<<<full_size / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres1, cuda_fullres1, 1, height, width);
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres1, full_size * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../out.npy", cpu_out, {1, 256, 256});
//
    cudaDeviceSynchronize();

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

    CHECK_CUDA_ERRS( cudaMemcpy(out, cuda_fullres1, full_size * sizeof(float), cudaMemcpyDeviceToHost) );

    std::chrono::steady_clock::time_point all_end = std::chrono::steady_clock::now();

    return std::make_pair(std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count(),
                          std::chrono::duration_cast<std::chrono::microseconds>(all_end - all_begin).count());
}

int main(int argc, char **argv) {
    char* input_path;
    char* output_path;
    int num_runs;

    ParseArguments(argc, argv, &input_path, &output_path, &num_runs);
    printf("Denoising %s and saving the result to %s\n", input_path, output_path);

    int width, height, channels;
    uint8_t* image = stbi_load(input_path, &width, &height, &channels, STBI_grey);
    if (channels != 1) {
        printf("Only grayscale images supported. Provided image has %d channels.\n", channels);
    }
    float* in_float_image = AllocateArray<float>(width, height, channels);
    float* out_float_image = AllocateArray<float>(width, height, channels);
    UintToFloat(image, in_float_image, width, height, channels, 1.f/255.f);

    int64_t total_duration_with = 0;
    int64_t total_duration_without = 0;
    for (int n_run = 0; n_run < num_runs; ++n_run) {
        auto [without_allocs, with_allocs] = DenoiseMNIST(in_float_image, out_float_image, height, width);
        std::cout << "Run " << n_run << std::endl;
        std::cout << "Took " << without_allocs << "[µs] to complete (without allocation)" << std::endl;
        std::cout << "Took " << with_allocs << "[µs] to complete (with allocation)" << std::endl;
        total_duration_without += without_allocs;
        total_duration_with += with_allocs;
    }
    std::cout << "Mean runtime over " << num_runs << " runs: " << total_duration_without / num_runs << "[µs] (without allocation)" << std::endl;
    std::cout << "Mean runtime over " << num_runs << " runs: " << total_duration_with / num_runs << "[µs] (with allocation)" << std::endl;


    FloatToUint(out_float_image, image, width, height, channels, 255.f);

    SaveImage(image, output_path, width, height, channels);

    return 0;
}
