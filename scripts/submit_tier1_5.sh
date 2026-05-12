#!/bin/bash
# Tier 1.5 submission: long-training ViT-S anchors for the paper.
#
# Jobs:
#   - A+VICReg @ 600ep ViT-S, 3 seeds
#   - A+SIGReg @ 600ep ViT-S, 3 seeds
#   - B+SIGReg @ 600ep ViT-S, 2 additional seeds
#   - B+VICReg @ 600ep ViT-S, 2 additional seeds
#
# This converts the high-variance 600ep Tier 1 result into a defensible table:
# A-cells anchor the B-vs-A claim at long training, while B-cell extra seeds
# stabilize the SIGReg-vs-VICReg comparison.
set -euo pipefail

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
MAX_PENDING=10
MAX_OWN_JOBS=18

JOBS=(
    "tin_600ep_vits_a_vicreg_proj 42 40:00:00"
    "tin_600ep_vits_a_vicreg_proj 123 40:00:00"
    "tin_600ep_vits_a_vicreg_proj 7 40:00:00"
    "tin_600ep_vits_a_sigreg_proj 42 40:00:00"
    "tin_600ep_vits_a_sigreg_proj 123 40:00:00"
    "tin_600ep_vits_a_sigreg_proj 7 40:00:00"
    "tin_600ep_vits_b_sigreg_proj 314 40:00:00"
    "tin_600ep_vits_b_sigreg_proj 2718 40:00:00"
    "tin_600ep_vits_b_vicreg_proj 314 40:00:00"
    "tin_600ep_vits_b_vicreg_proj 2718 40:00:00"
)

count_pending() {
    squeue -u "$USER" -h -t PD -o '%i' | wc -l | tr -d ' '
}

count_own_jobs() {
    squeue -u "$USER" -h -o '%i' | wc -l | tr -d ' '
}

submit_one() {
    local cfg="$1"
    local seed="$2"
    local tbudget="$3"
    local jname="${cfg}_s${seed}"
    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=${jname}
#SBATCH --output=${PROJECT_DIR}/logs/${jname}_%j.out
#SBATCH --error=${PROJECT_DIR}/logs/${jname}_%j.err
#SBATCH --time=${tbudget}
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=ALL
#SBATCH --exclude=watgpu1008,watgpu408,watgpu308

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${cfg}.yaml --seed ${seed}
EOF
}

echo "Tier 1.5 submission: ${#JOBS[@]} jobs queued. max_pending=${MAX_PENDING}, max_total=${MAX_OWN_JOBS}."

for entry in "${JOBS[@]}"; do
    read -r cfg seed tbudget <<< "$entry"

    while true; do
        pending=$(count_pending)
        total=$(count_own_jobs)
        if [ "$pending" -lt "$MAX_PENDING" ] && [ "$total" -lt "$MAX_OWN_JOBS" ]; then
            break
        fi
        echo "[queue gate] pending=${pending} total=${total} -- sleeping 60s"
        sleep 60
    done

    submit_one "$cfg" "$seed" "$tbudget"
    echo "Submitted: ${cfg}_s${seed} (time=${tbudget})"
    sleep 3
done

echo "Tier 1.5: all ${#JOBS[@]} jobs submitted."
