# BitonicSortCUDA

This project implements the Bitonic Sort algorithm using CUDA for parallel processing on NVIDIA GPUs. The implementation includes a CUDA kernel for the sorting algorithm and utility functions for generating and verifying the sorted array.

## Building the Project

To build the project, ensure you have CUDA installed and set the `CUDA_PATH` environment variable. Then, run the following command:
  
```bash
make
```

## Running the Project

```bash
./V1.exe <q> <p>
```

Where `<q>` and `<p>` are positive integers that determine the size of the array `N = 2^(q + p)`.
