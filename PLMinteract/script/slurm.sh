#!/bin/bash -l
#SBATCH --account=xxx(your own project account)
#SBATCH --job-name=test
#SBATCH --time=07-00:00:00 
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
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

# (1) PLM-interact training, 
# srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length $max_length --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 


# (2) Evaluate trained models (Validation and test trained models)
# Setting your own dev_filepath, test_filepath, epochs, output_filepath, resume_from_checkpoint (trained models), offline_model_path (ESM2 model)

# srun -u python predict_ddp.py --seed 2 --batch_size_val 32 --epochs 20 --dev_filepath $dev_filepath --test_filepath $test_filepath --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $max_length --offline_model_path $offline_model_path 


# (3) load the 650M_human_V11_PPI_model/35M_human_V11_PPI_model pytorch.bin from huggingface.
# Setting your own output filepath and resume_from_checkpoint (the download huggingface folder).
# srun -u python inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length $max_length --offline_model_path $offline_model_path


# (4) mutation effect training
resume_from_checkpoint=$resume_from_checkpoint
output='outpath/'
mkdir -p $output
task_name=taskname
maxlength=2196
input_data=input_data/
srun -u python mutation_train.py --epochs 50 --seed 2 --data mutation --task_name $task_name --batch_size_train 1 --batch_size_val 4 --train_filepath ${input_data}train_ppi_seqs.csv --dev_filepath ${input_data}val_ppi_seqs.csv  --output_path $output --warmup_steps 2000 --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $maxlength --gradient_accumulation_steps 32 


# (5)Mutation effects prediction
model_path='huggingface_download/'
resume_from_checkpoint=${model_path}pytorch_model.bin
output=${model_path}predict/
mkdir -p $output
maxlength=3833
input_data=input_data/
srun -u python mutation_predict.py --seed 2 --data mutation --batch_size_val 1 --test_filepath ${input_data}decreasing_increasing_test_PPIs.csv --output_path $output --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $maxlength 
