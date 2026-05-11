#!/bin/bash
# Tier 1 submission: 21 jobs total
#   1A: TinyIN no-proj B+SIG @ 600ep    (3 seeds, ~10h each)
#   1B: TinyIN ViT-Small 4-cell matrix @ 300ep (12 jobs, ~15h each)
#   1C: TinyIN ViT-Small B-cell @ 600ep (6 jobs,  ~30h each)
#
# Order: cheapest/most-informative first so we get signal as early as possible.
# 1A runs first (10h) — confirms or refutes projector confound at 600ep.
# 1B second (15h) — establishes ViT-S replication of main matrix.
# 1C last (30h) — extends scaling story to ViT-S.
set -euo pipefail

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
MAX_PENDING=10
MAX_OWN_JOBS=18

JOBS=(
    # 1A — small, fast, high value: closes one self-acknowledged gap
    "tin_600ep_b_sigreg_noproj 42 14:00:00"
    "tin_600ep_b_sigreg_noproj 123 14:00:00"
    "tin_600ep_b_sigreg_noproj 7 14:00:00"

    # 1B — ViT-Small 300ep matrix (cheaper than 1C, run first for early signal)
    "tin_300ep_vits_b_sigreg_proj 42 22:00:00"
    "tin_300ep_vits_b_sigreg_proj 123 22:00:00"
    "tin_300ep_vits_b_sigreg_proj 7 22:00:00"
    "tin_300ep_vits_b_vicreg_proj 42 22:00:00"
    "tin_300ep_vits_b_vicreg_proj 123 22:00:00"
    "tin_300ep_vits_b_vicreg_proj 7 22:00:00"
    "tin_300ep_vits_a_vicreg_proj 42 22:00:00"
    "tin_300ep_vits_a_vicreg_proj 123 22:00:00"
    "tin_300ep_vits_a_vicreg_proj 7 22:00:00"
    "tin_300ep_vits_a_sigreg_proj 42 22:00:00"
    "tin_300ep_vits_a_sigreg_proj 123 22:00:00"
    "tin_300ep_vits_a_sigreg_proj 7 22:00:00"

    # 1C — ViT-Small 600ep B-cell (longest runs, last to submit)
    "tin_600ep_vits_b_sigreg_proj 42 40:00:00"
    "tin_600ep_vits_b_sigreg_proj 123 40:00:00"
    "tin_600ep_vits_b_sigreg_proj 7 40:00:00"
    "tin_600ep_vits_b_vicreg_proj 42 40:00:00"
    "tin_600ep_vits_b_vicreg_proj 123 40:00:00"
    "tin_600ep_vits_b_vicreg_proj 7 40:00:00"
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
#SBATCH --exclude=watgpu1008,watgpu408

cd ${PROJECT_DIR}
python3 run_experiment.py --config ${CONFIGS_DIR}/${cfg}.yaml --seed ${seed}
EOF
}

echo "Tier 1 submission: ${#JOBS[@]} jobs queued. max_pending=${MAX_PENDING}, max_total=${MAX_OWN_JOBS}."

for entry in "${JOBS[@]}"; do
    read -r cfg seed tbudget <<< "$entry"

    while true; do
        pending=$(count_pending)
        total=$(count_own_jobs)
        if [ "$pending" -lt "$MAX_PENDING" ] && [ "$total" -lt "$MAX_OWN_JOBS" ]; then
            break
        fi
        echo "[queue gate] pending=${pending} total=${total} — sleeping 60s"
        sleep 60
    done

    submit_one "$cfg" "$seed" "$tbudget"
    echo "Submitted: ${cfg}_s${seed} (time=${tbudget})"
    sleep 3
done

echo "Tier 1: all ${#JOBS[@]} jobs submitted."
