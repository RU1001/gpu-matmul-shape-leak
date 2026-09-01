# Timing Side Channel: GPU Matmul Kernels Leak Tensor Shape 

Demonstrating that a GPU matrix multiplication kernel timing leaks the reduction dimension K via the k-loop iteration count.
This project was inspired by Paul C. Kocher's Paper:[Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems](https://paulkocher.com/doc/TimingAttacks.pdf)

## The Findings
associated case study:[How MatMul Kernel Timing Leaks Tensor Shape](https://ominaelsheikh.substack.com/p/how-matmul-kernel-timing-leaks-tensor)

## Repo structure
 
```
src/
  matmul_kernel.py       Triton blocked matmul, fixed config M,N,K (128x64x32)
 
experiments/
  shape_sweep.py         K-timing sweep (512..4096, step 8), shuffled order to mitigate unwanted output/influence from thermal drift
  analyze.py             Generates plots and summary CSV from raw timings
  data/
    timings.csv          Raw per-trial timings (449 K values x 100 trials x 2 stacks)
    summary.csv          Per-shape mean and std
  plots/
    timing_vs_shape.png  Main result figure
```
 
## Reproducing
 
Requires an NVIDIA GPU with CUDA support. Tested on a T4 (sm_75) via AWS EC2.
 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
 
# Quick sanity check (~30s)
python -m experiments.shape_sweep --dry-run
 
# Full sweep (~25 min on T4)
python -m experiments.shape_sweep
 
# Generate plots
python -m experiments.analyze
```
