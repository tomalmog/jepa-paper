#!/bin/bash
# Curvature ablation for B+SIG+proj TinyIN 300ep: c={0.25, 0.5, 2.0} × 3 seeds = 9 jobs.
# Baseline c=1.0 already complete (from follow-up matrix, mean 14.0%).
# Queue-gated, idempotent.

set -euo pipefail

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
MAX_PENDING=10
MAX_OWN_JOBS=18

JOBS=(
    "tin_300ep_b_sigreg_proj_c025 42 10:00:00"
    "tin_300ep_b_sigreg_proj_c025 123 10:00:00"
    "tin_300ep_b_sigreg_proj_c025 7 10:00:00"
    "tin_300ep_b_sigreg_proj_c05 42 10:00:00"
    "tin_300ep_b_sigreg_proj_c05 123 10:00:00"
    "tin_300ep_b_sigreg_proj_c05 7 10:00:00"
    "tin_300ep_b_sigreg_proj_c20 42 10:00:00"
    "tin_300ep_b_sigreg_proj_c20 123 10:00:00"
    "tin_300ep_b_sigreg_proj_c20 7 10:00:00"
)

count_pending() {
    squeue -u "$USER" -h -t PD -o '%i' | wc -l | tr -d ' '
}
count_own_jobs() {
    squeue -u "$USER" -h -o '%i' | wc -l | tr -d ' '
}
job_already_queued() {
    local jname="$1"
    squeue -u "$USER" -h -o '%j' | grep -Fx "$jname" > /dev/null
}
job_results_exist() {
    local cfg="$1"; local seed="$2"
    [ -f "$PROJECT_DIR/runs/${cfg}_seed${seed}/results.json" ]
}

submit_one() {
    local cfg="$1"; local seed="$2"; local tbudget="$3"
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

echo "Curvature ablation: ${#JOBS[@]} jobs in list. Will skip any already queued or completed."

submitted=0
skipped=0
for entry in "${JOBS[@]}"; do
    read -r cfg seed tbudget <<< "$entry"
    jname="${cfg}_s${seed}"

    if job_already_queued "$jname"; then
        echo "[skip] ${jname} already in queue"
        skipped=$((skipped+1))
        continue
    fi
    if job_results_exist "$cfg" "$seed"; then
        echo "[skip] ${jname} already has results.json"
        skipped=$((skipped+1))
        continue
    fi

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
    submitted=$((submitted+1))
    echo "Submitted: ${jname} (time=${tbudget})"
    sleep 3
done

echo "Done. submitted=${submitted}, skipped=${skipped}."
