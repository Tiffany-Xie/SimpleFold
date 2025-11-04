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
FORMAT=mmcif


# OUT_CSV <- results/<run name>/<run name>.csv
FASTA_DIR=""

usage() {
  echo "surprise surprise"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fasta_dir|-f) FASTA_DIR="${2:?}"; shift 2 ;;
#    --out_path|-o)  OUT_PATH="${2:?}"; shift 2 ;;
    --model|-m)     MODEL="${2:?}"; shift 2 ;;
    --backend|-b)   BACKEND="${2:?}"; shift 2 ;;
    --num_steps|-n) NSTEPS="${2:?}"; shift 2 ;;
    --tau|-t)       TAU="${2:?}"; shift 2 ;;
    --nsample|-s)   NSAMPLE="${2:?}"; shift 2 ;;
    --run_name|-r)  RUN_NAME="${2:?}"; shift 2 ;;
    --seed|-S)      SEED="${2:?}"; shift 2 ;;
    --format|-F)    FORMAT="${2:?}"; shift 2 ;;
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
echo "Output format: $FORMAT"
echo

mkdir -p "${OUT_PATH}"


##############
# simplefold prediction iteration
##############


simplefold \
  --simplefold_model "$MODEL" \
  --seed "$SEED" \
  --num_steps "$NSTEPS" --tau "$TAU" \
  --nsample_per_protein "$NSAMPLE" \
  --plddt \
  --fasta_path "$FASTA_DIR" \
  --output_dir "$OUT_PATH" \
  --output_format $FORMAT \
  --backend "$BACKEND" || true


echo "ALL DONE! Results -> $OUT_PATH"

cp -f "./Logs/simplefold_${SLURM_JOB_ID}.log" \
    "$MAIN/results/${RUN_NAME}/simplefold_${SLURM_JOB_ID}.log"






