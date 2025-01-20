# BitonicSortCUDA

By Tzanetis Savvas - 10889

## Introducion

This project implements the Bitonic Sort algorithm using **CUDA** for parallel processing on NVIDIA GPUs in the **C++** programming language. The primary objective is to sort a dataset of $N = 2^n$ numbers (where $n \in N$). The implementation employs parallel processing to achieve efficient sorting, making it suitable for large-scale data sets.

## Installation

To get started, clone the repository and install the necessary dependencies:

```bash
git  clone  https://github.com/stzanetis/BitonicSortCUDA.git
cd  BitonicSortCUDA
```

Ensure you have the following installed:

- A C++ compiler (gcc)
- CUDA Libraries
- Make

## Usage

In order to be able to test this implementation you will have to navigate to either of the `V0`, `V1`, `V2` versions directories, and build the project using `make`.

### Running the Project

There are three options for running the compiled code.

1. The first method seen below, used inside the selected version's directory generates a random array of $2^{27}$ integers and sorts them with the selected method `V0` ,`V1` or `V2`.

```bash
make run
```

2. For the second method below, you have to run the generated `.exe` file (for example V1.exe) and execute with a parameters $n$, where $2^{n}$ is the number of random integers to be sorted.

```bash
./V1.exe <n>
```
3. The final method required `python` to be installed, along with the `mathplotlib` library. Running the test script found inside the `Testing` directory, the code is compiled and executed automatically with $n$ values ranging between `[20:29]`. Finally the script, displays a graphic showing the execution times of each version compared to **qsort**.
```bash
cd ./Testing
./test.py
```

After testing you should clean any build artifacts by executing `make clean`.
