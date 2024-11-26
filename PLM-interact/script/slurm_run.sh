#!/bin/bash -l
#SBATCH --account=project0019 
#SBATCH --job-name=test
#SBATCH --time=07-00:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
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

# Train on PPIs
# srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length $max_length --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 

# Inference for PPIs
# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath /mnt/data/project0019/danliu/Data/basedata/dscript/output/output/ecoli.ppi.qrels.seq.test.csv --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath  --resume_from_checkpoint $resume_from_checkpoint --max_length $max_length --offline_model_path $offline_model_path 

# predict_ddp
# resume_from_checkpoint=/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/
# output_filepath=/users/2656169l/PLM-interact/PLM-interact/res_35m_v11/
# mkdir -p $output_filepath
# srun -u python /users/2656169l/PLM-interact/PLM-interact/predict_ddp.py --seed 2 --batch_size_val 32 --dev_filepath '/mnt/data/project0019/danliu/Data/basedata/dscript/output/output/human.ppi.qrels.seq.test.csv' --test_filepath '/mnt/data/project0019/danliu/Data/basedata/dscript/output/output/yeast.ppi.qrels.seq.test.csv' --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t12_35M_UR50D --embedding_size 480 --max_length 1603 --offline_model_path /mnt/data/project0019/danliu/PPI/offline/


# 35M_human_V11_PPI_model
# resume_from_checkpoint=/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/
# output_filepath=/users/2656169l/PLM-interact/PLM-interact/res_35m_v11_infer/
# mkdir -p $output_filepath 
# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath /mnt/data/project0019/danliu/Data/basedata/dscript/output/output/ecoli.ppi.qrels.seq.test.csv --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path /mnt/data/project0019/danliu/PPI/offline/


# 650M_human_V11_PPI_model
resume_from_checkpoint=/mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/
output_filepath=/users/2656169l/PLM-interact/PLM-interact/res_35m_v11_infer/
mkdir -p $output_filepath 
srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath /mnt/data/project0019/danliu/Data/basedata/dscript/output/output/ecoli.ppi.qrels.seq.test.csv --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path /mnt/data/project0019/danliu/PPI/offline/


/mnt/data/project0019/danliu/Data/basedata/dscript/output/output/human.ppi.qrels.seq.test.csv

# 650M_human_virus_PPI_model
# output_filepath=/users/2656169l/PLM-interact/PLM-interact/res_35m_v11/
# mkdir -p $output_filepath 
# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 4 --test_filepath /mnt/data/project0019/danliu/Data/basedata/dscript/output/output/ecoli.ppi.qrels.seq.test.csv --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint /mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/ --max_length 1600 --offline_model_path /mnt/data/project0019/danliu/PPI/offline/



# 650M_human_V12_PPI_model
# output_filepath=/users/2656169l/PLM-interact/PLM-interact/res_35m_v11/
# mkdir -p $output_filepath 
# srun -u python /users/2656169l/PLM-interact/PLM-interact/inference_PPI.py --seed 2 --batch_size_val 4 --test_filepath /mnt/data/project0019/danliu/Data/basedata/dscript/output/output/ecoli.ppi.qrels.seq.test.csv --model_name 'esm2_t12_35M_UR50D' --embedding_size 480 --output_filepath $output_filepath --resume_from_checkpoint /mnt/data/project0019/danliu/PPI/checkpoint/35m/dscript/task2/huggingface_model/ --max_length 1600 --offline_model_path /mnt/data/project0019/danliu/PPI/offline/

