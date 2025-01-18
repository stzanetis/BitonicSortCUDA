#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "../Utilities/utilities.h"

__global__ void bitonicSortStep(int *dev_values, int stage, int step) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int partner = tid ^ step;

    if (tid < partner) {
        if ((partner & stage) == 0) {
            if (dev_values[tid] > dev_values[partner]) {
                int temp = dev_values[tid];
                dev_values[tid] = dev_values[partner];
                dev_values[partner] = temp;
            }
        } else {
            if (dev_values[tid] < dev_values[partner]) {
                int temp = dev_values[tid];
                dev_values[tid] = dev_values[partner];
                dev_values[partner] = temp;
            }
        }
    }
}

void bitonicSort(int *values, int N) {
    int *dev_values;
    size_t size = N * sizeof(int);

    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    dim3 blocks((N + 511) / 512);
    dim3 threads(512);

    for (int stage = 2; stage <= N; stage <<= 1) {
        for (int step = stage >> 1; step > 0; step = step >> 1) {
            bitonicSortStep<<<blocks, threads>>>(dev_values, stage, step);
            cudaDeviceSynchronize();
        }
    }

    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
    cudaFree(dev_values);
}

int main(int argc, char *argv[]) {
    int q, p;
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " {q} {p}" << std::endl;
        return 1;
    } else {
        p = std::atoi(argv[1]);
        q = std::atoi(argv[2]);
    }

    int N = 1 << (q + p);   // N = 2^(q + p)
    int *values = new int[N];
    std::cout << "Number of elements: " << N << std::endl;

    generateArray(values, N);

    //printf("Unsorted array: ");
    //printArray(values, N);

    auto start = std::chrono::high_resolution_clock::now();
    bitonicSort(values, N);
    auto end = std::chrono::high_resolution_clock::now();

    //printf("Sorted array: ");
    //printArray(values, N);

    if (isSorted(values, N)) {
        std::cout << "The array is sorted correctly." << std::endl;
    } else {
        std::cout << "The array is NOT sorted correctly." << std::endl;
    }

    std::chrono::duration<double> duration = end - start;
    std::cout << "V0 Bitonic sort took " << duration.count() << " seconds." << std::endl;

    delete[] values;
    return 0;
}