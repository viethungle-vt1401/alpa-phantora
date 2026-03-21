# Phantora + Alpa Setup Guide (CS590 Project)

This guide provides **end-to-end instructions** to set up, build, and run Phantora on the Duke Slurm cluster using a **CPU-only, standalone configuration**.

# Overview

We modified Phantora to:

- Run without CUDA dependencies  
- Use CPU-only PyTorch  
- Disable FlashAttention and CUDA RPC  
- Run as a standalone simulator

# Initial Setup

Python environment setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Rust:
```bash
curl https://sh.rustup.rs -sSf | sh
source ~/.cargo/env
```

Build and test Phantora:
```bash
sbatch build_and_test.sh
```