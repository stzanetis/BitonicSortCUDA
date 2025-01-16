#include "utilities.cuh"
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