#!/bin/bash
# Submit all experiments to SLURM on the Waterloo cluster.
# Each job runs one config × one seed. We target H200s (watgpu-500/700/800).

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"

# Seeds for the 3-way replication
SEEDS="42 123 7"

# Models A, B, C — 3 seeds each = 9 jobs
for model in model_a model_b model_c; do
    for seed in $SEEDS; do
        JOB_NAME="${model}_s${seed}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${PROJECT_DIR}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/${JOB_NAME}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${model}.yaml --seed ${seed}
EOF
        echo "Submitted: ${JOB_NAME}"
    done
done

# Confounder tests — single seed (42) is enough to check the hypothesis
for confounder in confounder_lr2x confounder_lr4x confounder_normed; do
    JOB_NAME="${confounder}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${PROJECT_DIR}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/${JOB_NAME}_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${confounder}.yaml --seed 42
EOF
    echo "Submitted: ${JOB_NAME}"
done

echo ""
echo "Total jobs submitted: 12 (9 model×seed + 3 confounders)"
echo "Check status: squeue -u \$USER"
