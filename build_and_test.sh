#!/usr/bin/env bash
#SBATCH --job-name=phantora_test
#SBATCH --output=logs/test_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --partition=compsci

# Load modules
module load cuda

# Activate environments
source ~/.cargo/env
source ~/work/alpa-phantora/.venv/bin/activate

# Fix tch/PyTorch issues
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$LD_LIBRARY_PATH

cd ~/work/alpa-phantora/Phantora/phantora

cargo build --release

# Run binary
RUST_BACKTRACE=full ./target/release/simulator --netconfig ~/work/alpa-phantora/Phantora/netconfig.toml &

SIM_PID=$!

sleep 2
python -c "
import torch
x = torch.randn(1000,1000)
y = torch.mm(x,x)
print(y.shape)
"

wait $SIM_PID