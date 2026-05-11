#!/bin/bash
# Queue-gated submission for the 24-job follow-up matrix:
#   - 12 × tin_600ep (4 configs × 3 seeds)       time=20h
#   - 6  × tin_300ep noproj (2 configs × 3 seeds) time=10h
#   - 6  × tin_300ep_a_sigreg hparam debug         time=10h
#
# Submits jobs sequentially; waits whenever pending depth >= 10.

set -euo pipefail

PROJECT_DIR="$HOME/jepa-paper"
CONFIGS_DIR="$PROJECT_DIR/configs/experiments"
MAX_PENDING=10
MAX_OWN_JOBS=18   # total R+PD ceiling for our account

# Full list of (config_name, seed, walltime) tuples.
# Order: most-informative jobs first so they start running ASAP.
JOBS=(
    # B-cell 600ep first (the critical scaling points)
    "tin_600ep_b_sigreg_proj 42 20:00:00"
    "tin_600ep_b_sigreg_proj 123 20:00:00"
    "tin_600ep_b_sigreg_proj 7 20:00:00"
    "tin_600ep_b_vicreg_proj 42 20:00:00"
    "tin_600ep_b_vicreg_proj 123 20:00:00"
    "tin_600ep_b_vicreg_proj 7 20:00:00"
    # No-projector ablation (high value, fast)
    "tin_300ep_b_sigreg_noproj 42 10:00:00"
    "tin_300ep_b_sigreg_noproj 123 10:00:00"
    "tin_300ep_b_sigreg_noproj 7 10:00:00"
    "tin_300ep_b_vicreg_noproj 42 10:00:00"
    "tin_300ep_b_vicreg_noproj 123 10:00:00"
    "tin_300ep_b_vicreg_noproj 7 10:00:00"
    # A+SIGReg hparam debug (short runs)
    "tin_300ep_a_sigreg_proj_lowlr 42 10:00:00"
    "tin_300ep_a_sigreg_proj_lowlr 123 10:00:00"
    "tin_300ep_a_sigreg_proj_lowlr 7 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 42 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 123 10:00:00"
    "tin_300ep_a_sigreg_proj_lowwt 7 10:00:00"
    # A-cell 600ep last (least critical — A collapses on TinyIN anyway)
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

echo "Follow-up matrix: ${#JOBS[@]} jobs to submit. Max pending=${MAX_PENDING}, max total=${MAX_OWN_JOBS}."

for entry in "${JOBS[@]}"; do
    read -r cfg seed tbudget <<< "$entry"

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
    echo "Submitted: ${cfg}_s${seed} (time=${tbudget})"
    sleep 3
done

echo "All ${#JOBS[@]} jobs submitted."
