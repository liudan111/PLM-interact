#!/bin/bash -l
#SBATCH --account=project0019 
#SBATCH --job-name=test
#SBATCH --time=07-00:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=200G
#SBATCH --output=err/%x.%j.out
#SBATCH --error=err/%x.%j.err

source ~/.bashrc
conda activate sentence_pair

echo "NODELIST="${SLURM_NODELIST}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export JOB_ID=${SLURM_JOBID} # Set torchrun ID based on Slurm ID, avoid conflict between jobs.
export NUM_TRAINERS=${SLURM_NTASKS_PER_NODE}
export WORLD_SIZE=${SLURM_NTASKS}

# predict on ds data using input query, text data, output prediciton scores
# test_filepath=/users/2656169l/PPI/sentence_transformer/inference/binder_proteins_test.csv
# output_filepath=/users/2656169l/PPI/sentence_transformer/inference/predict/
# mkdir -p $output_filepath
# resume_from_checkpoint=/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/
# max_length=654

# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'facebook/esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath  --resume_from_checkpoint $resume_from_checkpoint --max_length $max_length


# Mutations analysis
# test_filepath=/users/2656169l/PPI/sentence_transformer/inference/binder_proteins_test.csv
# output_filepath=/users/2656169l/PPI/sentence_transformer/inference/predict/
# mkdir -p $output_filepath
# resume_from_checkpoint=/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/
# max_length=654
# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'facebook/esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath  --resume_from_checkpoint $resume_from_checkpoint --max_length $max_length