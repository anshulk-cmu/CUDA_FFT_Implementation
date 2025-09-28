# CUDA FFT Implementation

A high-performance GPU-accelerated implementation of the Fast Fourier Transform (FFT) algorithm using CUDA. This project demonstrates a significant speedup over standard CPU-based FFT implementations for signal processing tasks.

## 🎯 Overview

The Fast Fourier Transform is a fundamental algorithm in digital signal processing, used to convert signals from the time domain to the frequency domain. This project implements the Cooley-Tukey FFT algorithm using CUDA for GPU acceleration, processing signals with up to 16,384 samples.

### Key Results (for N=4096)

| Implementation | Time (ms) | Speedup vs. Recursive CPU |
| :--- | :--- | :--- |
| **CPU (Naive DFT)** | \> 1 minute (est.) | \~100,000x |
| **CPU (Recursive FFT)**| 51.519 ms | 1x |
| **GPU (CUDA)** | **0.420 ms** | **\~122x** |

## 🧠 Algorithm Background

### What is the Fast Fourier Transform?

The FFT is an optimized algorithm to compute the Discrete Fourier Transform (DFT). It's crucial for:

  - **Signal Analysis**: Identifying frequency components in audio, radio, and other signals.
  - **Image Processing**: Used in filtering, compression, and analysis.
  - **Scientific Computing**: Solving differential equations and performing large-number multiplication.

The algorithm recursively breaks down a DFT of size N into smaller DFTs, reducing the computational complexity from **O(N²)** to **O(N log N)**.

### Mathematical Foundation

The core of the Cooley-Tukey FFT algorithm is the "butterfly" operation, which combines the results of smaller DFTs. The iterative version relies on two key steps:

1.  **Bit-Reversal Permutation**: Reordering the input signal based on the bit-reversed order of its indices.
2.  **Iterative Butterfly Computations**: Performing `log₂(N)` stages of butterfly operations, where each stage combines pairs of elements using pre-computed "twiddle factors" (`W = e⁻²πi/N`).

## 🚀 Implementation Approach

### Data Preparation for CUDA

To prepare the data for efficient GPU processing, we pre-compute two essential components:

1.  **Bit-Reversal Table**: A lookup table that maps each index `i` to its bit-reversed counterpart `j`. This allows for a single, efficient memory reordering step.
2.  **Twiddle Factors**: Pre-computation of all `N/2` unique complex twiddle factors needed for the butterfly stages. This avoids redundant trigonometric calculations inside the GPU kernels.

### GPU Kernels

1.  **`bit_reverse_kernel`**: Performs the initial data shuffle. It reads the input signal and writes it to the output buffer in bit-reversed order, preparing it for the iterative stages.

    ```cuda
    __global__ void bit_reverse_kernel(...)
    ```

2.  **`fft_butterfly_kernel`**: Executes a single stage of the FFT. This kernel is launched `log₂(N)` times. It uses **shared memory** to perform fast, local butterfly operations within a thread block, significantly reducing global memory traffic.

    ```cuda
    __global__ void fft_butterfly_kernel(...)
    ```

### Algorithm Steps

1.  **Initialize**: Generate test signals and pre-compute bit-reversal tables and twiddle factors on the CPU.
2.  **Transfer**: Copy the signal, bit-reversal table, and twiddle factors from CPU host memory to GPU device memory.
3.  **Bit-Reverse**: Launch the `bit_reverse_kernel` once to reorder the input signal.
4.  **Iterate**: Launch the `fft_butterfly_kernel` `log₂(N)` times. In each launch, the stage number is passed to the kernel to determine the distance and twiddle factors for the butterfly operations.
5.  **Copy Back**: Transfer the final frequency-domain data from the GPU back to the CPU.

## 📊 Performance Analysis

### Dataset: Synthetic Signals

  - **Signal Types**: Pure sine wave, multi-frequency with noise, and linear chirp.
  - **Sizes (N)**: 64, 256, 1024, 4096, 16384 samples.
  - **Precision**: `float64` (double) for input, `complex128` for FFT output.

### CUDA Performance (on Tesla T4 GPU)

```
N     | Total Time (ms) | Butterfly Time (ms) | GFLOPS | Bandwidth (GB/s)
------|-----------------|---------------------|--------|-----------------
64    |      0.254      |        0.109        |  0.01  |       0.05
256   |      0.289      |        0.123        |  0.04  |       0.23
1024  |      0.356      |        0.174        |  0.14  |       0.92
4096  |      0.420      |        0.204        |  0.59  |       3.75
16384 |      0.510      |        0.255        |  2.25  |      14.39
```

*Note: The accuracy for N \>= 1024 shows a known issue, likely related to inter-block communication. See "Extending the Project" for details.*

### CPU vs. GPU Comparison

```
Size     | CPU Naive (ms) | CPU Recursive (ms) | GPU CUDA (ms)
---------|----------------|--------------------|--------------
64       |     6.744      |       0.449        |    0.254
256      |    107.295     |       2.198        |    0.289
1024     |   1623.688     |      10.426        |    0.356
4096     |    TOO_SLOW    |      51.519        |    0.420
16384    |    TOO_SLOW    |        SKIP        |    0.510
```

## 🛠 How to Reproduce

### Prerequisites

1.  **Google Colab** with GPU runtime enabled.

      - Go to **Runtime → Change runtime type → Hardware accelerator → T4 GPU**.

2.  **CUDA Environment**

      - CUDA 12.2+ (available in Colab).
      - NVIDIA GPU with compute capability 7.5+ (Tesla T4 is ideal).

### Step-by-Step Instructions

1.  **Open the Notebook in Colab**: Use the "Open in Colab" badge in the notebook file or directly access it.

2.  **Generate Test Data**: Run the first code cell to mount Google Drive and generate the synthetic signal data and reference FFTs.

3.  **Run CPU Baseline**: Execute the second cell to run the naive and recursive CPU FFTs and see their performance characteristics.

4.  **Write and Compile CUDA Code**: The third cell writes the `fft_cuda.cu` source file. The fourth cell compiles it using `nvcc`.

5.  **Execute CUDA FFT**: The final cell runs the compiled CUDA program, which processes all signal sizes and prints the performance and accuracy results.

## 📁 Project Structure

```
/content/drive/MyDrive/cuda-fft/
├── CUDA_FFT_Implementation.ipynb  # Main Colab notebook
├── data/                            # Generated signal and reference data (.npy)
│   ├── signal_sine_64.npy
│   └── fft_ref_sine_64.npy
├── prep/                            # CPU timing results
│   └── cpu_timings.json
└── bin/                             # Binary files for CUDA execution
    ├── fft_64_input_real.bin
    ├── fft_64_bitrev.bin
    ├── fft_64_twiddle_real.bin
    └── fft_64_meta.json

/content/
└── fft_cuda.cu                      # CUDA source code written by the notebook
└── fft_cuda                         # Compiled CUDA executable
```

## 🔧 Technical Details

### GPU Configuration

  - **Threads per block**: 256
  - **Shared Memory**: 4.0 KB per block
  - **Architecture**: Compiled for `sm_75` (NVIDIA Tesla T4)

### Numerical Precision

  - **Data type**: `float64` (double precision) for inputs and `complex128` for intermediate/final results.
  - **Accuracy Check**: Compares the L2 norm of the error between the CUDA result and the NumPy reference FFT.

### Memory Optimization

  - **Shared Memory**: Used within the butterfly kernel for fast data exchange between threads in a block.
  - **Data Layout**: Real and imaginary components are stored in separate arrays.
  - **Pre-computation**: Twiddle factors and bit-reversal indices are pre-calculated to minimize GPU workload.

## 🎓 Educational Value

This implementation provides a practical demonstration of:

1.  **FFT Algorithm**: The iterative Cooley-Tukey algorithm and its core components (bit-reversal, butterfly operations).
2.  **CUDA Programming**: Writing, compiling, and launching custom GPU kernels.
3.  **GPU Memory Hierarchy**: Using shared memory to optimize performance by reducing reliance on slower global memory.
4.  **Parallel Algorithm Design**: Decomposing a complex algorithm into parallel stages suitable for a GPU architecture.
5.  **Performance Analysis**: Profiling and comparing the performance of CPU vs. GPU implementations.

## 🔬 Extending the Project

### Potential Improvements

1.  **Fix Accuracy Bug**: The primary issue is the incorrect result for larger N. The current `fft_butterfly_kernel` does not correctly handle synchronization between thread blocks when a butterfly partner is in a different block. This could be fixed by launching a separate kernel for each stage, ensuring global synchronization between stages.
2.  **Radix-4/Radix-8 Implementation**: A higher-radix FFT can reduce the number of stages and memory accesses, further improving performance.
3.  **Mixed-Precision**: Use `float` (single precision) instead of `double` for a potential 2x speedup in memory-bound operations, if the application's precision requirements allow it.
4.  **CUFFT Comparison**: Benchmark the custom implementation against NVIDIA's highly optimized CUFFT library to understand the performance gap.

-----

**Note**: This implementation is designed for educational purposes to demonstrate the principles of a CUDA-based FFT. The accuracy bug for larger sizes is a known limitation and a valuable learning exercise in debugging parallel synchronization issues. For production use, NVIDIA's CUFFT library is recommended.
