#include <iostream>
#include <chrono>
#include "../Utilities/utilities.h"

int compare(const void *a, const void *b) {
    return (*(int*)a - *(int*)b);
}

void qsort(int *values, int N) {
    std::qsort(values, N, sizeof(int), compare);
}

int main(int argc, char *argv[]) {
    int N;
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <n>" << std::endl;
        return 1;
    } else {
        N = 1 << (std::atoi(argv[1]));
    }

    int *values = new int[N];
    std::cout << "Number of elements to be sorted: " << N << std::endl;

    generateArray(values, N);

    auto start = std::chrono::high_resolution_clock::now();
    qsort(values, N);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    std::cout << "Qsort sort took " << duration.count() << " seconds." << std::endl;

    delete[] values;
    return 0;
}