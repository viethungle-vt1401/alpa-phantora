#!/usr/bin/env bash
#SBATCH --job-name=phantora_eval
#SBATCH --output=logs/eval_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64gb                   # Job memory request
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH --partition=compsci-gpu
#SBATCH --gres=gpu:4
#SBATCH --no-requeue
#SBATCH --open-mode=append

# === DEFINE ABSOLUTE PATHS ===
BASE_DIR="$HOME/work/alpa-phantora"

module load cuda
source ~/.cargo/env

echo "==================================================="
echo " Starting Phantora + Alpa Evaluation Suite"
echo " Job ID: $SLURM_JOB_ID"
echo " All results will be logged to this main output file."
echo "==================================================="

# ==========================================
# 0. BUILD RUST (Using CPU Environment)
# ==========================================
cd $BASE_DIR/Phantora/phantora
echo "Building Phantora Simulator..."
(
    source $BASE_DIR/.venv-gpu/bin/activate
    export LIBTORCH_USE_PYTORCH=1
    export LIBTORCH_BYPASS_VERSION_CHECK=1
    export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
    export PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
    export LD_LIBRARY_PATH=$TORCH_LIB:$PYTHON_LIB:$LD_LIBRARY_PATH
    
    cargo build --release
)

# If the build fails, kill the script so we don't run broken tests
if [ $? -ne 0 ]; then
    echo "CRITICAL ERROR: Cargo build failed! Aborting evaluations."
    exit 1
fi

run_framework_eval() {
    local FRAMEWORK=$1
    local TEST_SCRIPT=$2

    echo "---------------------------------------------------"
    echo " Evaluating $FRAMEWORK"
    echo "---------------------------------------------------"

    # ==========================================
    # A. REAL GPU RUN (Using GPU Environment)
    # ==========================================
    echo "[1/2] Running Real Physical GPUs Baseline..."
    (
        source $BASE_DIR/.venv-gpu/bin/activate
        export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
        export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
        export PYTHONWARNINGS="ignore"

        unset PHANTORA 
        cd $BASE_DIR/Phantora/tests
        
        START_TIME=$(date +%s%3N)
        # Output flows directly to SLURM log
        CUDA_LAUNCH_BLOCKING=1 torchrun --nproc_per_node=4 $TEST_SCRIPT --num_layers 4 --hidden_size 1024 --micro_batch_size 1 --sequence_length 1024
        END_TIME=$(date +%s%3N)
        
        REAL_DURATION=$((END_TIME - START_TIME))
        echo "Real Run Completed in ${REAL_DURATION} ms"
    ) 

    # ==========================================
    # B. PHANTORA SIMULATION (Using CPU Environment)
    # ==========================================
    echo "[2/2] Running Phantora Simulation..."
    (
        source $BASE_DIR/.venv/bin/activate
        export LIBTORCH_USE_PYTORCH=1
        export LIBTORCH_BYPASS_VERSION_CHECK=1
        export TORCH_LIB=$(python -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
        export PYTHON_LIB=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))")
        export LD_LIBRARY_PATH=$TORCH_LIB:$PYTHON_LIB:$LD_LIBRARY_PATH
        
        export PHANTORA=1 
        export PHANTORA_SOCKET_PREFIX="/tmp/phantora"
        
        export PYTHONWARNINGS="ignore"
        
        cd $BASE_DIR/Phantora/tests
        
        START_TIME=$(date +%s%3N)
        
        # Run the simulator with 4 distributed CPU processes
        torchrun --nproc_per_node=4 $TEST_SCRIPT --num_layers 4 --hidden_size 1024 --micro_batch_size 1 --sequence_length 1024
        
        END_TIME=$(date +%s%3N)
        
        SIM_OVERHEAD=$((END_TIME - START_TIME))
        echo "Simulation Overhead (Search Time): ${SIM_OVERHEAD} ms"
    )
}

# Execute the Suite
run_framework_eval "DeepSpeed" "test_deepspeed.py"
run_framework_eval "Megatron" "test_megatron.py"

echo "==================================================="
echo " Evaluation Suite Complete."
echo "==================================================="