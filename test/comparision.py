import time
import numpy as np
import scipy.sparse as sp
import cupy as cp
import cupyx.scipy.sparse as cpx_sparse
import matplotlib.pyplot as plt

# -----------------------------
# Experiment Parameters
# -----------------------------
sizes = [2000, 4000, 6000]   # matrix sizes
density = 0.001              # sparsity level (0.1% non-zero entries)

cpu_times, gpu_times, hybrid_times = [], [], []

for N in sizes:
    print(f"\n🔹 Running for {N}x{N} matrix...")

    # Generate sparse matrices (CPU)
    A_cpu = sp.random(N, N, density=density, format="csr", dtype=np.float32)
    B_cpu = sp.random(N, N, density=density, format="csr", dtype=np.float32)

    # ---------------- CPU Execution ----------------
    start = time.time()
    C_cpu = A_cpu @ B_cpu
    cpu_time = time.time() - start
    cpu_times.append(cpu_time)
    print(f"CPU time: {cpu_time:.4f} sec")

    # ---------------- GPU Execution ----------------
    A_gpu = cpx_sparse.csr_matrix(A_cpu)
    B_gpu = cpx_sparse.csr_matrix(B_cpu)

    cp.cuda.Stream.null.synchronize()
    start = time.time()
    C_gpu = A_gpu @ B_gpu
    cp.cuda.Stream.null.synchronize()
    gpu_time = time.time() - start
    gpu_times.append(gpu_time)
    print(f"GPU time: {gpu_time:.4f} sec")

    # ---------------- Hybrid Execution ----------------
    rows = N // 2

    # CPU handles top half
    start = time.time()
    C_cpu_half = A_cpu[:rows] @ B_cpu

    # GPU handles bottom half
    A_half_gpu = cpx_sparse.csr_matrix(A_cpu[rows:])
    C_gpu_half = A_half_gpu @ B_gpu
    cp.cuda.Stream.null.synchronize()

    hybrid_time = time.time() - start
    hybrid_times.append(hybrid_time)
    print(f"Hybrid time: {hybrid_time:.4f} sec")

# -----------------------------
# Plot Results
# -----------------------------
plt.figure(figsize=(8,6))
plt.plot(sizes, cpu_times, marker='o', label="CPU Only (SciPy)")
plt.plot(sizes, gpu_times, marker='o', label="GPU Only (CuPy)")
plt.plot(sizes, hybrid_times, marker='o', label="Hybrid (CPU+GPU)")
plt.xlabel("Matrix Size (N x N)")
plt.ylabel("Execution Time (seconds)")
plt.title("Sparse Matrix Multiplication: CPU vs GPU vs Hybrid")
plt.legend()
plt.grid(True)
plt.show()
