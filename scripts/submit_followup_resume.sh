#!/bin/bash
# Resume follow-up matrix submission — only submits jobs whose name is NOT already
# in the queue (handles daemon restart safely). Queue-gated: max pending=10.

set -euo pipefail

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
MAX_PENDING=10
MAX_OWN_JOBS=18

JOBS=(
    "tin_600ep_b_sigreg_proj 42 20:00:00"
    "tin_600ep_b_sigreg_proj 123 20:00:00"
    "tin_600ep_b_sigreg_proj 7 20:00:00"
    "tin_600ep_b_vicreg_proj 42 20:00:00"
    "tin_600ep_b_vicreg_proj 123 20:00:00"
    "tin_600ep_b_vicreg_proj 7 20:00:00"
    "tin_300ep_b_sigreg_noproj 42 10:00:00"
    "tin_300ep_b_sigreg_noproj 123 10:00:00"
    "tin_300ep_b_sigreg_noproj 7 10:00:00"
    "tin_300ep_b_vicreg_noproj 42 10:00:00"
    "tin_300ep_b_vicreg_noproj 123 10:00:00"
    "tin_300ep_b_vicreg_noproj 7 10:00:00"
    "tin_300ep_a_sigreg_proj_lowlr 42 10:00:00"
    "tin_300ep_a_sigreg_proj_lowlr 123 10:00:00"
    "tin_300ep_a_sigreg_proj_lowlr 7 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 42 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 123 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 7 10:00:00"
    "tin_600ep_a_vicreg_proj 42 20:00:00"
    "tin_600ep_a_vicreg_proj 123 20:00:00"
    "tin_600ep_a_vicreg_proj 7 20:00:00"
    "tin_600ep_a_sigreg_proj 42 20:00:00"
    "tin_600ep_a_sigreg_proj 123 20:00:00"
    "tin_600ep_a_sigreg_proj 7 20:00:00"
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
    # If this seed has already produced results.json, skip.
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

echo "Follow-up matrix resume: ${#JOBS[@]} jobs in list. Will skip any already queued or completed."

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

    # Wait until queue has room
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
