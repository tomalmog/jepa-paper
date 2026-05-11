#!/bin/bash
# Submit the SIGReg experimental matrix: {A,B} × {VICReg,SIGReg} × {CIFAR-10, CIFAR-100, Tiny-ImageNet}
# 3 seeds each = 4 × 3 × 3 = 36 jobs

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
SEEDS="42 123 7"

# Per-dataset time budgets (wall time was ~6h for TinyIN C+VICReg 300ep on H100-class)
# CIFAR-10 50ep ~45 min → 2h with headroom
# CIFAR-100 200ep ~4h → 6h with headroom
# Tiny-ImageNet 300ep ~6h → 10h with headroom

declare -A TIMES=(
    ["c10_50ep"]="02:00:00"
    ["c100_200ep"]="06:00:00"
    ["tin_300ep"]="10:00:00"
)

# Configs to submit. Format: dataset_prefix config_name
CONFIGS=(
    "c10_50ep c10_50ep_a_vicreg_proj"
    "c10_50ep c10_50ep_a_sigreg_proj"
    "c10_50ep c10_50ep_b_vicreg_proj"
    "c10_50ep c10_50ep_b_sigreg_proj"
    "c100_200ep c100_200ep_a_vicreg_proj"
    "c100_200ep c100_200ep_a_sigreg_proj"
    "c100_200ep c100_200ep_b_vicreg_proj"
    "c100_200ep c100_200ep_b_sigreg_proj"
    "tin_300ep tin_300ep_a_vicreg_proj"
    "tin_300ep tin_300ep_a_sigreg_proj"
    "tin_300ep tin_300ep_b_vicreg_proj"
    "tin_300ep tin_300ep_b_sigreg_proj"
)

for entry in "${CONFIGS[@]}"; do
    read -r prefix cfg <<< "$entry"
    time_budget=${TIMES[$prefix]}
    for seed in $SEEDS; do
        JOB_NAME="${cfg}_s${seed}"
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${JOB_NAME}
#SBATCH --output=${PROJECT_DIR}/logs/${JOB_NAME}_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/${JOB_NAME}_%j.err
#SBATCH --time=${time_budget}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${cfg}.yaml --seed ${seed}
EOF
        echo "Submitted: ${JOB_NAME} (time=${time_budget})"
        sleep 3  # space out submissions to avoid hammering scheduler
    done
done

echo "All 36 jobs submitted."
