#!/bin/bash -l
#SBATCH --account=project0019 
#SBATCH --job-name=test
#SBATCH --time=07-00:00:00 
#SBATCH --nodes=8
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpu-a100-80
#SBATCH --gres=gpu:4
#SBATCH --mem=500G
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


srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length 2146 --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 