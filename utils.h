#ifndef UTILS_H
#define UTILS_H

#include <cstdlib>
#include <iostream>
#include <cmath>
#include <chrono>

#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

/// Wrapper function for saving images
/// @param[in] src                      Source 8-bit image
/// @param     path                     Path to save
/// @param     width, height, channels  Image dimensions
void SaveImage(const uint8_t* src, const char* path, int width, int height, int channels) {
    int comp = STBI_rgb;
    if (channels == 1) {
        comp = STBI_grey;
    }
    stbi_write_png(path, width, height, comp, src, width * channels);
}

template<typename T>
T* AllocateArray(int width, int height, int channels=1) {
    size_t image_size = width * height * channels;
    T* image = static_cast<T*>(calloc(image_size, sizeof(T)));
    return image;
}

void UintToFloat(const uint8_t* src, float* dst, int width, int height, int channels, float mul=1.f, float add=0.f) {
    size_t total_len = width * height * channels;
    while (total_len--) {
        *dst++ = static_cast<float>(*src++) * mul + add;
    }
}

void FloatToUint(const float* src, uint8_t* dst, int width, int height, int channels, float mul=1.f, float add=0.f) {
    size_t total_len = width * height * channels;
    while (total_len--) {
        *dst++ = static_cast<uint8_t>(std::min(std::max(mul * *src++ + add, 0.f), 255.f));
    }
}

void ParseArguments(int argc, char** argv, char** input, char** output, int* num_runs) {
    const char help_str[] = "MNIST denoising via CNN autoencoder\n"
                            "usage: %s input_image output_image --benchmark num_runs\n\n"
                            "Options:\n"
                            "  --benchmark\t(optional) number of runs for benchmarking\n\n"
                            "Help:\n"
                            "  -h\t\tShow this text\n";
    if (argc < 3 || argc == 4 || (argc > 1 && strcmp(argv[1], "-h") == 0)) {
        printf(help_str, argv[0]);
        exit(0);
    }
    *input = argv[1];
    *output = argv[2];
    if (argc == 5) {
        if (strcmp(argv[3], "--benchmark") == 0) {
            *num_runs = strtol(argv[4], nullptr, 10);
        } else {
            printf("Unknown option: %s\n", argv[3]);
            exit(0);
        }
    } else {
        *num_runs = 1;
    }
}

#endif //UTILS_H
