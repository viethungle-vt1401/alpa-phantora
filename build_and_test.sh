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
source .venv/bin/activate

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

# === DYNAMIC BATCH JOB CONFIGURATION ===
# Use the unique SLURM Job ID to create a unique filename for this run.
# If SLURM_JOB_ID is not set (e.g., running the script manually outside of sbatch), 
# it defaults to 'local_' + the shell's process ID.
JOB_ID=${SLURM_JOB_ID:-local_$$}
GRAPH_FILENAME="compute_graphs/compute_graph_${JOB_ID}.json"

# 6. Extract the computational graph from PyTorch
echo "Building computational graph -> ${GRAPH_FILENAME}..."
python build_graph.py \
    --output ${GRAPH_FILENAME} \
    --batch-size 256 \
    --hidden-size 2048

# 7. Run the Rust simulator with the uniquely generated graph
echo "Starting Phantora Simulator with ${GRAPH_FILENAME}..."
RUST_BACKTRACE=full ./target/release/simulator \
    --netconfig ~/work/alpa-phantora/Phantora/netconfig.toml \
    --graph ${GRAPH_FILENAME} &

# 8. Capture the simulator's process ID and wait for it to finish
SIM_PID=$!
wait $SIM_PID

# 9. Clean up the JSON file to save disk space on the cluster
# rm ${GRAPH_FILENAME}
echo "Cleaned up ${GRAPH_FILENAME}."

echo "Simulation complete for Job ${JOB_ID}."