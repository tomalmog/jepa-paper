#!/bin/bash
# Resubmit the remaining SIGReg matrix jobs, excluding watgpu1008 (no torch).
# The already-running jobs (1430202–7, 1430214, 1430220, 1430221) are skipped.

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"

declare -A TIMES=(
    ["c10_50ep"]="02:00:00"
    ["c100_200ep"]="06:00:00"
    ["tin_300ep"]="10:00:00"
)

# Format: "config_prefix config_name seed"
JOBS=(
    # CIFAR-10 A (all 6 needed)
    "c10_50ep c10_50ep_a_vicreg_proj 42"
    "c10_50ep c10_50ep_a_vicreg_proj 123"
    "c10_50ep c10_50ep_a_vicreg_proj 7"
    "c10_50ep c10_50ep_a_sigreg_proj 42"
    "c10_50ep c10_50ep_a_sigreg_proj 123"
    "c10_50ep c10_50ep_a_sigreg_proj 7"
    # CIFAR-100 A (all 6 needed)
    "c100_200ep c100_200ep_a_vicreg_proj 42"
    "c100_200ep c100_200ep_a_vicreg_proj 123"
    "c100_200ep c100_200ep_a_vicreg_proj 7"
    "c100_200ep c100_200ep_a_sigreg_proj 42"
    "c100_200ep c100_200ep_a_sigreg_proj 123"
    "c100_200ep c100_200ep_a_sigreg_proj 7"
    # CIFAR-100 B+VICReg (s42 1430214 is running, need s123 + s7)
    "c100_200ep c100_200ep_b_vicreg_proj 123"
    "c100_200ep c100_200ep_b_vicreg_proj 7"
    # CIFAR-100 B+SIGReg (all 3 needed)
    "c100_200ep c100_200ep_b_sigreg_proj 42"
    "c100_200ep c100_200ep_b_sigreg_proj 123"
    "c100_200ep c100_200ep_b_sigreg_proj 7"
    # Tiny-ImageNet A+VICReg (s42 running as 1430220, s123 running as 1430221, need s7)
    "tin_300ep tin_300ep_a_vicreg_proj 7"
    # Tiny-ImageNet A+SIGReg (all 3 needed)
    "tin_300ep tin_300ep_a_sigreg_proj 42"
    "tin_300ep tin_300ep_a_sigreg_proj 123"
    "tin_300ep tin_300ep_a_sigreg_proj 7"
    # Tiny-ImageNet B+VICReg (all 3 needed)
    "tin_300ep tin_300ep_b_vicreg_proj 42"
    "tin_300ep tin_300ep_b_vicreg_proj 123"
    "tin_300ep tin_300ep_b_vicreg_proj 7"
    # Tiny-ImageNet B+SIGReg (all 3 needed)
    "tin_300ep tin_300ep_b_sigreg_proj 42"
    "tin_300ep tin_300ep_b_sigreg_proj 123"
    "tin_300ep tin_300ep_b_sigreg_proj 7"
)

for entry in "${JOBS[@]}"; do
    read -r prefix cfg seed <<< "$entry"
    time_budget=${TIMES[$prefix]}
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
#SBATCH --exclude=watgpu1008

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${cfg}.yaml --seed ${seed}
EOF
    echo "Submitted: ${JOB_NAME} (time=${time_budget})"
    sleep 3
done

echo "Done. $(( ${#JOBS[@]} )) jobs submitted."
