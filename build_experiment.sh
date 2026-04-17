#!/usr/bin/env bash
#SBATCH --job-name=phantora_scaling
#SBATCH --output=logs/test_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=00:30:00
#SBATCH --partition=compsci

# --- 1. CONFIGURATION (Edit these before sbatch) ---
TARGET_ALPHA="0.1"
TARGET_BETA="1e-9"
TARGET_MEM_GB="16"
TARGET_GPUS="16"  # <--- Set this to 4, 8, or 16 for your scaling tests
# ----------------------------------------------------

# Load modules and setup environment
module load cuda
source ~/.cargo/env
source .venv/bin/activate

# Fix library linking
export LIBTORCH_USE_PYTORCH=1
export LIBTORCH_BYPASS_VERSION_CHECK=1
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
export LD_LIBRARY_PATH=$TORCH_LIB:$PYTHON_LIB:$LD_LIBRARY_PATH

# Move to working directory
cd ~/work/alpa-phantora/Phantora/phantora

# --- 2. DYNAMIC SOURCE MODIFICATION ---
# Update mod.rs for cost parameters
MOD_RS="phantora/src/alpa/mod.rs"
sed -i "s/const ALPHA_COMM: f64 = .*/const ALPHA_COMM: f64 = $TARGET_ALPHA;/" $MOD_RS
sed -i "s/const BETA_COMM: f64 = .*/const BETA_COMM: f64 = $TARGET_BETA;/" $MOD_RS
sed -i "s/const DEVICE_MEMORY_LIMIT_BYTES: f64 = .*/const DEVICE_MEMORY_LIMIT_BYTES: f64 = $TARGET_MEM_GB.0 * 1024.0 * 1024.0 * 1024.0;/" $MOD_RS

# Update main.rs for GPU count
# This replaces "let num_gpus = 4;" with your target value
MAIN_RS="phantora/src/main.rs"
sed -i "s/let num_gpus = .*/let num_gpus = $TARGET_GPUS;/" $MAIN_RS

# --- 3. PARALLEL-SAFE COMPILATION ---
JOB_ID=${SLURM_JOB_ID:-local_$$}
UNIQUE_BIN="./target/release/simulator_${JOB_ID}"

echo "Compiling unique simulator for Job ${JOB_ID} with ${TARGET_GPUS} GPUs..."
cargo build --release
cp ./target/release/simulator $UNIQUE_BIN

# --- 4. GRAPH GENERATION ---
GRAPH_FILENAME="compute_graphs/compute_graph_${JOB_ID}.json"
python build_graph.py --output ${GRAPH_FILENAME} --batch-size 256 --hidden-size 2048

# --- 5. TIMED EXECUTION ---
START_TIME=$(date +%s%3N)

$UNIQUE_BIN \
    --netconfig ~/work/alpa-phantora/Phantora/netconfig.toml \
    --graph ${GRAPH_FILENAME}

END_TIME=$(date +%s%3N)
SEARCH_LATENCY=$((END_TIME - START_TIME))

echo "----------------------------------------------------"
echo "RUN CONFIG: GPUs=${TARGET_GPUS}, Alpha=${TARGET_ALPHA}, Beta=${TARGET_BETA}"
echo "Search Time (Optimization Latency): ${SEARCH_LATENCY} ms"
echo "----------------------------------------------------"

# Clean up
rm ${GRAPH_FILENAME}
rm ${UNIQUE_BIN}