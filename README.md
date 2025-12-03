# GTSAM Scaling Limits: Large-Scale Bundle Adjustment Benchmark

## 1. Project Introduction
This project investigates the scalability limits of **GTSAM (Georgia Tech Smoothing and Mapping)** when applied to large-scale Structure-from-Motion (SfM) problems. As modern datasets grow to thousands of images and millions of points, the computational bottleneck in Bundle Adjustment (BA) shifts from simple graph construction to the factorization of massive linear systems.

We evaluate the performance of direct solvers (Multifrontal Cholesky) across diverse topologies—ranging from sparse city-scale reconstructions to dense, object-centric scenes. Our experiments reveal that **graph connectivity (Schur complement density)**, rather than the raw number of parameters, is the primary predictor of memory exhaustion (OOM) failures.

To validate these findings, we utilize the standard **BAL** benchmark and preprocess raw **1DSfM** datasets using a custom COLMAP pipeline to generate valid, large-scale initializations for optimization.

---

## 2. Related Work & Context
Bundle Adjustment is the gold standard for refining 3D reconstructions, but its scalability is a longstanding challenge in computer vision and robotics.

### The Foundation of Scalable BA
The cornerstone of modern large-scale optimization is the exploitation of sparsity. **Bundle Adjustment in the Large (BAL)** [1] demonstrated that exact solutions for massive systems are feasible by leveraging the specific block-sparse structure of the Jacobian. Similarly, **Sparse Sparse Bundle Adjustment (SSBA)** [6] introduced efficient parallelized algorithms to accelerate the solving process by strictly avoiding operations on zero-blocks.

### Solving Strategies: Direct vs. Iterative
While direct methods (like Cholesky factorization used in early solvers) provide exact steps, they suffer from "fill-in"—where sparse matrices become dense during factorization. 
* **Divide-and-Conquer:** To mitigate this, approaches like [5] and **DeepLM** [4] decompose the problem into clusters (stochastic domain decomposition) or optimize global separators first.
* **Approximation:** For SLAM-like planar graphs, **Carlone et al.** [2] proposed fast approximations to bypass the full non-linear optimization.
* **Distributed Optimization:** Recent works have pushed for decentralization using ADMM to split the problem across systems, though this often requires complex hyper-parameter tuning as noted in **STBA** [58].

### The Data Challenge: BAL & 1DSfM
Scaling Bundle Adjustment requires solving two distinct types of complexity, which we address using two complementary datasets:

1.  **The Mathematical Challenge (BAL):** Derived from "Bundle Adjustment in the Large" [1], these datasets provide the "pure" optimization benchmark. They are pre-initialized and clean, allowing us to isolate the performance of the linear solver against massive, dense Hessian matrices without the noise of bad feature matching. They test the raw **computational limits** (RAM/FLOPs) of the solver.

2.  **The Robustness Challenge (1DSfM):**
    Real-world internet photo collections [7] introduce irregular graph topologies and high outlier ratios. Unlike the structured density of BAL, datasets like *Piccadilly* (linear street scenes) create drift-prone, band-diagonal sparsity patterns, while object-centric scenes like *Notre Dame* create dense Schur complements. This tests the **numerical stability** and **robustness** of the optimization engine against irregular, noisy data.

### 3. Hardware Constraints
* **RAM:** 32GB System Memory (Testing OOM boundaries).
* **Swap:** Configured 32GB swap space  for aggressive swapping (`vm.swappiness=100`).

### References
1. **S. Agarwal, N. Snavely, S. M. Seitz, and R. Szeliski.** "Bundle Adjustment in the Large." *ECCV*, 2010.
2. **L. Carlone, R. Aragues, J. A. Castellanos, and B. Bona.** "A fast and accurate approximation for planar pose graph optimization." *Intl. J. of Robotics Research*, 33(7):965-987, 2014.
3. **A. Jurić et al.** "A Comparison of Graph Optimization Approaches for Pose Estimation in SLAM."
4. **DeepLM:** "Large-scale Nonlinear Least Squares on Deep Learning Frameworks using Stochastic Domain Decomposition."
5. **Decentralization and Acceleration Enables Large-Scale Bundle Adjustment.**
6. **K. Konolige.** "Sparse sparse bundle adjustment." *BMVC*, 2010.
7. **K. Wilson and N. Snavely.** "Robust global translations with 1dsfm." *ECCV*, 2014.
