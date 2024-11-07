#!/bin/bash -l
#SBATCH --account=project0019 
#SBATCH --job-name=test
#SBATCH --time=07-00:00:00 
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpuplus
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


# hold_out_path=/mnt/data/project0019/danliu/PPI/virushostDB/identity40_holdout/Retroviridae/
# output_filepath='/mnt/data/project0019/danliu/PPI/checkpoint/35m/test/'
# srun -u python train_binary.py --epochs 30 --seed 2 --data ho --task_name 'test' --batch_size_train 1 --batch_size_val 32 --train_filepath ${hold_out_path}pairs_seqs_human_virus_train_2194.tsv --dev_filepath ${hold_out_path}pairs_seqs_human_virus_val_2211.tsv --test_filepath ${hold_out_path}pairs_seqs_human_virus_test_2166.tsv --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 4  --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --max_length 2000 --evaluation_steps 10 --sub_samples 5000 --offline_model_path '/mnt/data/project0019/danliu/PPI/offline/'


# hold_out_path=/mnt/data/project0019/danliu/PPI/virushostDB/identity40_holdout/Retroviridae/
# output_filepath='/mnt/data/project0019/danliu/PPI/checkpoint/35m/test/'
# srun -u python train_mlm.py --epochs 30 --seed 2 --data ho --task_name '1vs10' --batch_size_train 1 --train_filepath ${hold_out_path}pairs_seqs_human_virus_train_2194.tsv --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 4  --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --max_length 2000 --evaluation_steps 4 --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path '/mnt/data/project0019/danliu/PPI/offline/' 

train_filepath=/mnt/data/project0019/danliu/Data/basedata/dscript/output/output/
output_filepath='/mnt/data/project0019/danliu/PPI/checkpoint/35m/test/'
resume_from_checkpoint='/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task6/'
offline_model_path='/mnt/data/project0019/danliu/PPI/offline/' 
srun -u python inference_ddp.py --seed 2 --data 'test' --epochs 10 --task_name 'inference' --batch_size_val 16 --dev_filepath ${train_filepath}human.ppi.qrels.seq.test.csv --test_filepath ${train_filepath}worm.ppi.qrels.seq.test.csv --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1000 --offline_model_path $offline_model_path