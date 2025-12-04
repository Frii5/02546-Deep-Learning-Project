#!/bin/bash
#BSUB -q gpua40                  # queue/partition (e.g., GPU queue)
#BSUB -J tokenizer_train         # job name
#BSUB -n 8                       # number of CPU cores
#BSUB -gpu "num=1:mode=shared"   # request 1 GPU (shared mode)
#BSUB -W 24:00                   # wall-time limit (HH:MM)
#BSUB -R "rusage[mem=32GB]"      # memory requirement
#BSUB -R "span[hosts=1]"         # all cores on single host
#BSUB -o logs/train_%J.out       # stdout file, %J = job id
#BSUB -e logs/train_%J.err       # stderr file

# Load modules/environment
module load cuda/12.8.1
# module load python/3.10
# source activate my_env        
module load python3/3.11.9
source $BLACKHOLE/venv_tokenizers/bin/activate

python3.11 train_tokenizers.py

# Go to working directory
cd $BLACKHOLE/tokenization_project

# Run training script
python train_tokenizers.py
