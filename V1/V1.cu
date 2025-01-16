#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
//#include "../Utilities/utilities.cuh"
#include <cstdlib>
#include <ctime>

// Check if the array is sorted
bool isSorted(int *values, int N) {
    for (int i = 1; i < N; i++) {
        if (values[i - 1] > values[i]) {
            return false;
        }
    }
    return true;
}

// Initialize the array with random integers
void generateArray(int *values, int N) {
    std::srand(std::time(0));

    for (int i = 0; i < N; i++) {
        values[i] = std::rand() % 100; // 0 - 99
    }
}

__global__ void bitonicSortKernel(int *dev_values, int N) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;

    for (unsigned int k = 2; k <= N; k <<= 1) {
        for (unsigned int j = k >> 1; j > 0; j >>= 1) {
            unsigned int ixj = i ^ j;

            if (ixj > i) {
                if ((i & k) == 0) {
                    if (dev_values[i] > dev_values[ixj]) {
                        int temp = dev_values[i];
                        dev_values[i] = dev_values[ixj];
                        dev_values[ixj] = temp;
                    }
                } else {
                    if (dev_values[i] < dev_values[ixj]) {
                        int temp = dev_values[i];
                        dev_values[i] = dev_values[ixj];
                        dev_values[ixj] = temp;
                    }
                }
            }
            __syncthreads();
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

    bitonicSortKernel<<<blocks, threads>>>(dev_values, N);
    cudaDeviceSynchronize();

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

    auto start = std::chrono::high_resolution_clock::now();
    bitonicSort(values, N);
    auto end = std::chrono::high_resolution_clock::now();

    if (isSorted(values, N)) {
        std::cout << "The array is sorted correctly." << std::endl;
    } else {
        std::cout << "The array is NOT sorted correctly." << std::endl;
    }

    std::chrono::duration<double> duration = end - start;
    std::cout << "V1 Bitonic sort took " << duration.count() << " seconds." << std::endl;

    delete[] values;
    return 0;
}