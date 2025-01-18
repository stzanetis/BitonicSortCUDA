# BitonicSortCUDA

This project implements the Bitonic Sort algorithm using **CUDA** for parallel processing on NVIDIA GPUs in the **C++** programming language. The primary objective is to sort a dataset of $N = 2^q$ numbers (where $q \in N$). The implementation employs parallel processing to achieve efficient sorting, making it suitable for large-scale data sets.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/stzanetis/BitonicSortCUDA.git
cd BitonicSortCUDA
```

Ensure you have the following installed:

- A C++ compiler (gcc)
- CUDA Libraries
- Make

## Usage

In order to be able to test this implementation you will have to navigate to either of the `V0`, `V1`, `V2` versions directories, and build the project using `make`.

### Running the Project

There are two options for running the compiled code.
The method below, used inside the selected version's directory generates a random array of $2^{27}$ integers and sorts them with the selected method `V0` ,`V1` or `V2`.

```bash
make run
```

For the second method below, you have to run the generated `.exe` file (for example V1.exe) and execute with a parameters $n$, where $2^{n}$ is the number of random integers to be sorted.

```bash
./V1.exe <n>
```
