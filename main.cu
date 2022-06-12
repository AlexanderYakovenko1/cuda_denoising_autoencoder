#include <iostream>
#include <vector>

#include "include/cnpy.h"
#include "utils.h"
#include "layers.cuh"

constexpr int BLOCK_SIZE = 32;
constexpr int MAX_THREADS = 1024;

void DenoiseMNIST(const float* in, float* out, int height, int width, const std::string& weights_dir="../model_weights") {

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

    float* cuda_fullres1 = nullptr;
    float* cuda_fullres32_1 = nullptr;
    float* cuda_fullres32_2 = nullptr;
    float* cuda_halfres32_1 = nullptr;
    float* cuda_halfres32_2 = nullptr;
    float* cuda_quarterres32 = nullptr;

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres1, height * width * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_fullres1, in, height * width * sizeof(float), cudaMemcpyHostToDevice) );
//    cnpy::npy_save("../__in.npy", in, {1, 28, 28});

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres32_1, 32 * height * width * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_fullres32_2, 32 * height * width * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_halfres32_1, 32 * height / 2 * width / 2 * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_halfres32_2, 32 * height / 2 * width / 2 * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_quarterres32, 32 * height / 4 * width / 4 * sizeof(float)) );

//
    // todo: remore, this is for debug only
    float* cpu_out = static_cast<float *>(malloc(32 * height * width * sizeof(float)));
//

    dim3 numBlocksFull(width / BLOCK_SIZE + 1, height / BLOCK_SIZE + 1);
    dim3 numBlocksHalf(width / 2 / BLOCK_SIZE + 1, height / 2 / BLOCK_SIZE + 1);
    dim3 numBlocksQuarter(width / 4 / BLOCK_SIZE + 1, height / 4 / BLOCK_SIZE + 1);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    Conv2D_3x3<<<numBlocksFull, threadsPerBlock>>>(
            cuda_fullres1, cuda_fullres32_1, cuda_conv1_weight, cuda_conv1_bias, 1, 32, height, width);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error1: %s\n", cudaGetErrorString(error)); exit(-1);
    }

//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres32_1, 32 * height * width * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../0_conv1.npy", cpu_out, {32, 28, 28});
//

    ReLU<<<32 * height * width / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres32_1, cuda_fullres32_1);
//
//    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres32, 32 * height * width * sizeof(float), cudaMemcpyDeviceToHost) );
//    cnpy::npy_save("../1_relu1.npy", cpu_out, {32, 28, 28});
//

    MaxPool2D<<<numBlocksHalf, threadsPerBlock>>>(cuda_fullres32_1, cuda_halfres32_1, 32, height, width, 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error2: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_1, 32 * height / 2 * width / 2 * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../1_maxpool1.npy", cpu_out, {32, 14, 14});
//

    Conv2D_3x3<<<numBlocksHalf, threadsPerBlock>>>(
            cuda_halfres32_1, cuda_halfres32_2, cuda_conv2_weight, cuda_conv2_bias, 32, 32, height / 2, width / 2);
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_2, 32 * height / 2 * width / 2 * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../2_conv2.npy", cpu_out, {32, 14, 14});
//
    ReLU<<<32 * height / 2 * width / 2 / MAX_THREADS + 1, MAX_THREADS>>>(cuda_halfres32_2, cuda_halfres32_2);

    MaxPool2D<<<numBlocksHalf, threadsPerBlock>>>(cuda_halfres32_2, cuda_quarterres32, 32, height / 2, width / 2, 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error3: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_quarterres32, 32 * height / 4 * width / 4 * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../3_maxpool2.npy", cpu_out, {32, 7, 7});
//

    TranposedConv2D_3x3_2<<<numBlocksHalf, threadsPerBlock>>>(
            cuda_quarterres32, cuda_halfres32_1, cuda_halfres32_2, cuda_upconv1_weight, cuda_upconv1_bias, 32, 32, height / 4, width / 4);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error4: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_halfres32_1, 32 * height / 2 * width / 2 * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../4_upconv1.npy", cpu_out, {32, 14, 14});
//
    ReLU<<<32 * height / 2 * width / 2 / MAX_THREADS + 1, MAX_THREADS>>>(cuda_halfres32_1, cuda_halfres32_1);

    TranposedConv2D_3x3_2<<<numBlocksFull, threadsPerBlock>>>(
            cuda_halfres32_1, cuda_fullres32_1, cuda_fullres32_2, cuda_upconv2_weight, cuda_upconv2_bias, 32, 32, height / 2, width / 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error5: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres32_1, 32 * height * width * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../5_upconv2.npy", cpu_out, {32, 28, 28});
//
    ReLU<<<32 * height * width / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres32_1, cuda_fullres32_1);

    Conv2D_3x3<<<numBlocksFull, threadsPerBlock>>>(
            cuda_fullres32_1, cuda_fullres1, cuda_conv3_weight, cuda_conv3_bias, 32, 1, height, width);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error6: %s\n", cudaGetErrorString(error)); exit(-1);
    }
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres1, height * width * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../6_conv3.npy", cpu_out, {1, 28, 28});
//
    Sigmoid<<<height * width / MAX_THREADS + 1, MAX_THREADS>>>(cuda_fullres1, cuda_fullres1);
//
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_fullres1, height * width * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../out.npy", cpu_out, {1, 28, 28});
//
}

int main(int argc, char **argv) {

    cnpy::NpyArray img = cnpy::npy_load("../img.npy");
    float* loaded_data = img.data<float>();
    std::cout << img.shape[0] << img.num_vals << std::endl;
    std::cout << loaded_data[0] << std::endl;

    DenoiseMNIST(loaded_data, nullptr, 28, 28);

    return 0;


    cnpy::NpyArray weight1 = cnpy::npy_load("../model_weights/conv1.weight.npy");
    cnpy::NpyArray bias1 = cnpy::npy_load("../model_weights/conv1.bias.npy");


    float* cuda_in = nullptr;
    float* cuda_out = nullptr;
    float* cpu_out = static_cast<float *>(malloc(32 * img.num_vals * sizeof(float)));

    float* cuda_out2 = nullptr;
    float* cpu_out2 = static_cast<float *>(malloc(32 * img.num_vals / 2 / 2 * sizeof(float)));

    float* cuda_weight1 = nullptr;
    float* cuda_bias1 = nullptr;
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_in, img.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_out, 32 * img.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_out2, 32 * img.num_vals / 2 / 2 * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_in, loaded_data, img.num_vals * sizeof(float), cudaMemcpyHostToDevice) );

    CHECK_CUDA_ERRS( cudaMalloc(&cuda_weight1, weight1.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_weight1, weight1.data<float>(), weight1.num_vals * sizeof(float), cudaMemcpyHostToDevice) );
    CHECK_CUDA_ERRS( cudaMalloc(&cuda_bias1, bias1.num_vals * sizeof(float)) );
    CHECK_CUDA_ERRS( cudaMemcpy(cuda_bias1, bias1.data<float>(), bias1.num_vals * sizeof(float), cudaMemcpyHostToDevice) );

    dim3 numBlocks(img.shape[1] / BLOCK_SIZE + 1, img.shape[2] / BLOCK_SIZE + 1);
    printf("%d %d\n", numBlocks.x, numBlocks.y);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    Conv2D_3x3<<<numBlocks, threadsPerBlock>>>(cuda_in, cuda_out, cuda_weight1, cuda_bias1, 1, 32, 28, 28);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error1: %s\n", cudaGetErrorString(error)); exit(-1);
    }

    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_out, 32 * img.num_vals * sizeof(float), cudaMemcpyDeviceToHost) );

    cnpy::npy_save("../out1.npy", cpu_out, {32, 28, 28});

    ReLU<<<32 * img.num_vals / MAX_THREADS + 1, MAX_THREADS>>>(cuda_out, cuda_out);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error2: %s\n", cudaGetErrorString(error)); exit(-1);
    }
    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out, cuda_out, 32 * img.num_vals * sizeof(float), cudaMemcpyDeviceToHost) );
    cnpy::npy_save("../relu1.npy", cpu_out, {32, 28, 28});

    numBlocks = dim3(img.shape[1] / 2 / BLOCK_SIZE + 1, img.shape[2] / 2 / BLOCK_SIZE + 1);
    printf("%d %d\n", numBlocks.x, numBlocks.y);
    MaxPool2D<<<numBlocks, threadsPerBlock>>>(cuda_out, cuda_out2, 32, 28, 28, 2);
    error = cudaGetLastError();
    if(error != cudaSuccess) {
        printf("CUDA error3: %s\n", cudaGetErrorString(error)); exit(-1);
    }

    CHECK_CUDA_ERRS( cudaMemcpy(cpu_out2, cuda_out2, 32 * img.num_vals / 2 / 2 * sizeof(float), cudaMemcpyDeviceToHost) );

    cnpy::npy_save("../maxp1.npy", cpu_out2, {32, 14, 14});
}
