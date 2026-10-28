# Timing Side Channel: GPU Matmul Kernels Leak Tensor Shape 

Demonstrating that a GPU matrix multiplication kernel timing leaks the reduction dimension K via the k-loop iteration count.
This project was inspired by Paul C. Kocher's Paper:[Timing Attacks on Implementations of Diffie-Hellman, RSA, DSS, and Other Systems](https://paulkocher.com/doc/TimingAttacks.pdf)

## The Findings
associated case study: [How MatMul Kernel Timing Leaks Tensor Shape](https://ominaelsheikh.substack.com/p/how-matmul-kernel-timing-leaks-tensor)

## Repo structure
 
```
src/
  matmul_kernel.py       Triton blocked matmul, fixed config M,N,K (128x64x32)
 
experiments/
  shape_sweep.py         K-timing sweep (512..4096, step 8)
  analyze.py             Generates plots and summary CSV from raw timings
  data/
    timings.csv          Raw per-trial timings (449 K values × 15 trials × 2 implementations), where each trial is the mean over 20 timed launches.
    summary.csv          Per-shape mean, std, and trial count
  plots/
    timing_vs_shape.png result figure
```
 
## Reproducing
 
Requires an NVIDIA GPU with CUDA support. Tested on a T4 (sm_75) via AWS EC2.
 
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
 
# sanity check on small set of trials
python -m experiments.shape_sweep --dry-run
 
# Full sweep 
python -m experiments.shape_sweep
 
# Generate plots
python -m experiments.analyze
```
