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

# 4. Fix Python (pyo3) dynamic linking issues
export PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH=$PYTHON_LIB:$LD_LIBRARY_PATH

# 5. Move to working directory and compile
cd ~/work/alpa-phantora/Phantora/phantora
cargo build --release

# 6. Extract the computational graph from PyTorch
echo "Building computational graph..."
python build_graph.py

# 7. Run the Rust simulator with the generated graph in the background
echo "Starting Phantora Simulator..."
RUST_BACKTRACE=full ./target/release/simulator \
    --netconfig ~/work/alpa-phantora/Phantora/netconfig.toml \
    --graph compute_graph.json &

# 8. Capture the simulator's process ID and wait for it to finish
SIM_PID=$!
wait $SIM_PID

echo "Simulation complete."