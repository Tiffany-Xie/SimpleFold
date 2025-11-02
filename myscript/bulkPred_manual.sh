#!/bin/bash
#SBATCH --job-name=simplefold
#SBATCH --time=72:00:00
#SBATCH --partition=gen_gpu
#SBATCH --gres=gpu:NVIDIA_H100_NVL
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --tmp=300G
#SBATCH --output=./Logs/simplefold_%A.log

set -euo pipefail
MAIN="/hpf/projects/mtyers/ningrui/SimpleFold"

##############
# defaults
##############
RUN_NAME="monomer_test"

MODEL="simplefold_100M" #100M, 360M, 700M, 1.1B, 1.6B, 3B
BACKEND="torch" # mlx (apple stuff)
NSTEPS="500" # number of inference steps for flow-matching
TAU="0.01" # stochasticity scale
NSAMPLE="1"
SEED=17

# TARGET_PATH <- csv file containing PDB ID column
# OUT_CSV <- results/<run name>/<run name>.csv
TARGET_PATH=""
FASTA_DIR=""

usage() {
  echo "surprise surprise"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target|-t)    TARGET_PATH="${2:?}"; shift 2 ;;
    --fasta_dir|-f) FASTA_DIR="${2:?}"; shift 2 ;;
#    --out_path|-o)  OUT_PATH="${2:?}"; shift 2 ;;
    --model|-m)     MODEL="${2:?}"; shift 2 ;;
    --backend|-b)   BACKEND="${2:?}"; shift 2 ;;
    --num_steps|-n) NSTEPS="${2:?}"; shift 2 ;;
    --tau|-T)       TAU="${2:?}"; shift 2 ;;
    --nsample|-s)   NSAMPLE="${2:?}"; shift 2 ;;
    --run_name|-r)  RUN_NAME="${2:?}"; shift 2 ;;
    --seed|-S)      SEED="${2:?}"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "(x) Unknown flag: $1"; usage; exit 2 ;;
  esac
done

OUT_PATH="${MAIN}/results/${RUN_NAME}"

echo "Params loaded"

##############
# env + miscellaneous
##############

CONDA_BASE=/hpf/projects/mtyers/ningrui/miniconda3
source ${CONDA_BASE}/bin/activate ${CONDA_BASE}/envs/simplefold
echo "Env activated"

echo
echo "Run name: $RUN_NAME"
echo "Model: $MODEL"
echo "Backend: $BACKEND"
echo "n steps: $NSTEPS"
echo "tau: $TAU"
echo "n sample: $NSAMPLE"
echo "Seed: $SEED"
echo

mkdir -p "${OUT_PATH}"
mkdir -p "${OUT_PATH}/metrics"


OUT_CSV="${OUT_PATH}/${RUN_NAME}.csv"
if [[ ! -f "$OUT_CSV" ]]; then
  echo "pdb_id,model,backend,num_steps,tau,nsample,start_epoch,end_epoch,elapsed_sec,max_rss_kb,exit_code,job_id" > "$OUT_CSV"
fi

echo "Getting PDB IDs from target csv..."
mapfile -t PDB_IDS < <(python - "$TARGET_PATH" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as f:
    r = csv.DictReader(f)
    for row in r:
        pdb = row.get('PDB ID')
        if pdb:
            print(str(pdb).strip())
PY
)
printf '%s\n' "${PDB_IDS[@]}"

##############
# simplefold prediction iteration
##############

for PDB_ID in "${PDB_IDS[@]}"; do
  echo "Starting $PDB_ID Prediction..."

  if [[ -f "${FASTA_DIR}/${PDB_ID}.fasta" ]]; then
      FASTA_PATH="${FASTA_DIR}/${PDB_ID}.fasta"
  else
      echo "WARN: FASTA not found for ${PDB_ID}"
      echo "${PDB_ID},${MODEL},${BACKEND},${NSTEPS},${TAU},${NSAMPLE},,,,,,${SLURM_JOB_ID:-}" >> "$OUT_CSV"
      continue
  fi

  metrics_file="${OUT_PATH}/metrics/${PDB_ID}.txt"
  start_epoch="$(date +%s)"
  
  /usr/bin/time -f "elapsed_sec=%e\nmax_rss_kb=%M\nexit=%x" -o "$metrics_file" \
  simplefold \
    --simplefold_model "$MODEL" \
    --seed "$SEED" \
    --num_steps "$NSTEPS" --tau "$TAU" \
    --nsample_per_protein "$NSAMPLE" \
    --plddt \
    --fasta_path "$FASTA_PATH" \
    --output_dir "$OUT_PATH" \
    --backend "$BACKEND" || true

  end_epoch="$(date +%s)"

  elapsed="$(grep -E '^elapsed_sec=' "$metrics_file" | cut -d= -f2 || echo "")"
  rss_kb="$(grep -E '^max_rss_kb=' "$metrics_file" | cut -d= -f2 || echo "")"
  exit_code="$(grep -E '^exit=' "$metrics_file" | cut -d= -f2 || echo "1")"

  echo "${PDB_ID},${MODEL},${BACKEND},${NSTEPS},${TAU},${NSAMPLE},${start_epoch},${end_epoch},${elapsed},${rss_kb},${exit_code},${SLURM_JOB_ID:-}" >> "$OUT_CSV"
  echo "Done ${PDB_ID}: Time used=${elapsed}s, RSS=${rss_kb} KB"
  echo "-----------------------------------------------------------"

done

echo "ALL DONE! Results -> $OUT_CSV"






