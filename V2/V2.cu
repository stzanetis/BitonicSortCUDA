#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include "../Utilities/utilities.h"

__global__ void bitonicSortStep(int *dev_values, int threads, int stage, int step) {
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int partner = tid^step;
    
    if (partner > tid) {
        bool minmax = (tid & stage) == 0;
        if (minmax ? dev_values[tid] > dev_values[partner] : dev_values[tid] < dev_values[partner]) {
            int temp = dev_values[tid];
            dev_values[tid] = dev_values[partner];
            dev_values[partner] = temp;
        }
    } else {
        tid += threads;
        partner += threads;

        bool minmax = (tid & stage) == 0;
        if (minmax ? dev_values[tid] < dev_values[partner] : dev_values[tid] > dev_values[partner]) {
            int temp = dev_values[tid];
            dev_values[tid] = dev_values[partner];
            dev_values[partner] = temp;
        }
    }
}

__global__ void localSort(int *dev_values, int N, int stage, int step) {
    extern __shared__ int shared_values[];
    unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int local_tid = threadIdx.x;
    unsigned int offset = N >> 1;

    shared_values[local_tid] = dev_values[tid];
    shared_values[local_tid + min(offset, blockDim.x)] = dev_values[tid + offset];
    __syncthreads();

    if (tid < offset) {
        do {
            while (step > 0) {
                unsigned int partner = tid ^ step;
                unsigned int local_partner = partner - blockIdx.x * blockDim.x;
                
                if (partner > tid) {                    
                    if((tid & stage) == 0 ? shared_values[local_tid] > shared_values[local_partner] : shared_values[local_tid] < shared_values[local_partner]) {
                        int temp = shared_values[local_tid];
                        shared_values[local_tid] = shared_values[local_partner];
                        shared_values[local_partner] = temp;
                    }

                } else {
                    tid += offset;
                    partner += offset;
                    local_tid += min(offset, blockDim.x);
                    local_partner += min(offset, blockDim.x);

                    if((tid & stage) == 0 ? shared_values[local_tid] < shared_values[local_partner] : shared_values[local_tid] > shared_values[local_partner]) {
                        int temp = shared_values[local_tid];
                        shared_values[local_tid] = shared_values[local_partner];
                        shared_values[local_partner] = temp;
                    }

                    tid -= offset;
                    local_tid -= min(offset, blockDim.x);
                }
                step >>= 1;
                __syncthreads();
            }
            stage <<= 1;
            step = stage >> 1;
        } while (stage <= min(N, 1 << 10));
    }
    dev_values[tid] = shared_values[local_tid];
    dev_values[tid + offset] = shared_values[local_tid + min(offset, blockDim.x)];
    __syncthreads();
}

void bitonicSort(int *values, int N) {
    int *dev_values;
    size_t size = N * sizeof(int);
    int threads = N/2;
    int threadsPerBlock = 1024;
    int blocks = (threads - 1) / threadsPerBlock + 1;

    cudaMalloc((void**)&dev_values, size);
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

    localSort<<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(int)>>>(dev_values, N, 2, 1);

    for (int stage = 2048; stage <= N; stage <<= 1) {
        for (int step = stage >> 1; step > 512; step >>= 1) {
            bitonicSortStep<<<blocks, threadsPerBlock, threadsPerBlock * sizeof(int)>>>(dev_values, threads, stage, step);
            cudaDeviceSynchronize();
        }
        localSort<<<blocks, threadsPerBlock, 2 * threadsPerBlock * sizeof(int)>>>(dev_values, N, stage, 512);
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

    int N = 1 << (q + p);
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
    std::cout << "V2 Bitonic sort took " << duration.count() << " seconds." << std::endl;

    delete[] values;
    return 0;
}