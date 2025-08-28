# Concurrent Gaussian Elimination

## Introduction
This project implements Gaussian elimination with parallelization using **C++20** and **CUDA**.  
It explores two approaches:
- A **theoretical Foata Normal Form (FNF)-based method**, which explicitly models dependencies between operations.
- A **simplified parallel method**, which avoids heavy dependency graph construction but achieves the same effect in practice.

## Theoretical Background
Gaussian elimination solves systems of linear equations by transforming the matrix into an upper triangular form.  
Each elimination step involves:
1. Computing multipliers (how much of one row should be subtracted from another).
2. Multiplying row elements by the multiplier.
3. Subtracting the result from the target row.

While the classical algorithm executes these operations in sequence, many of them are independent and can run concurrently.  
The FNF framework groups independent operations into levels, showing which steps can be parallelized safely.  
Although elegant, explicitly constructing the dependency graph and Foata form becomes impractical for large matrices.

## Implementation
The project consists of:
- **Foata-based algorithm** – builds a dependency graph, derives Foata Normal Form, and executes each level in parallel on the GPU.
- **Simplified algorithm** – computes all multipliers for a pivot row in parallel, then performs all row updates in parallel, stage by stage. This avoids the overhead of graph construction.

Both versions run on the GPU with CUDA. The simplified version is much faster for larger systems, while the Foata version demonstrates the theoretical foundation.

## Usage
Build and run on Linux:
```bash
mkdir build
cmake -B./build -S.
cd build
make
./gauss
````

Requirements:

* `nvidia-cuda-toolkit`
* A CUDA-capable GPU

Input should be placed in `input.txt` in the project root.
After execution, the following files will be generated:

* `output-foata.txt` – solution using the FNF method
* `output-gaussian.txt` – solution using the simplified method
* `graph.dot` – dependency graph (visualizable with [Graphviz](https://graphviz.org/))

## Results

Both methods compute correct solutions. Small numerical differences may appear, but they do not affect correctness.

Performance comparison:

* For small systems (e.g. `N=4`), the FNF method is orders of magnitude slower due to extra graph construction and output.
* For larger systems (e.g. `N=100`), computing dependencies for FNF already takes longer than solving the system directly.
* The simplified method runs efficiently on GPU and scales well.

## Conclusion

* **FNF-based elimination**: showcases the theoretical approach to concurrency with explicit dependency handling.
* **Simplified elimination**: practical, much faster, and suitable for large systems.
* The project demonstrates both the *theory* and the *practice* of concurrent Gaussian elimination.
