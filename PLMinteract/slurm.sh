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

# Note : Please set the correct directory paths and parameters before running the following commands. We recommend running them one by one. Also, adjust the node settings in the SLURM script to match your HPC system.


# (1) PLM-interact training, Training PPI models using mask and binary classification losses.
max_length=1603
train_filepath=../cross_species_benchmarking/train/human.ppi.qrels.seq.train.csv
offline_model_path=../offline/
output_filepath=../checkpoint/run_plminteract/train_mlm/
mkdir -p $output_filepath
srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length $max_length --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 

# (2) PLM-interact training, Training PPI models using only binary classification loss.
max_length=1603
train_filepath=../cross_species_benchmarking/train/human.ppi.qrels.seq.train.csv
dev_filepath=../cross_species_benchmarking/val/human.ppi.qrels.seq.test.csv
test_filepath=../cross_species_benchmarking/test/ecoli.ppi.qrels.seq.test.csv
offline_model_path=../offline/
output_filepath=../checkpoint/run_plminteract/train_binary/
mkdir -p $output_filepath
srun -u python train_binary.py --epochs 10 --seed 2 --data 'human_V11' --task_name 'binary' --batch_size_train 1 --batch_size_val 32 --train_filepath $train_filepath --dev_filepath $dev_filepath --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length $max_length --offline_model_path $offline_model_path --evaluation_steps 10000 --sub_samples 10000


# (3) Evaluate trained models (Validation and test trained models)
# Setting your own dev_filepath, test_filepath, epochs, output_filepath, resume_from_checkpoint (trained models), offline_model_path (ESM2 model)
offline_model_path=../offline/
dev_filepath=../cross_species_benchmarking/val/human.ppi.qrels.seq.test.csv
test_filepath=../cross_species_benchmarking/test/ecoli.ppi.qrels.seq.test.csv
resume_from_checkpoint=../checkpoint/35m/
output_filepath=../checkpoint/run_plminteract/predict_ddp/
mkdir -p $output_filepath
max_length=1603
srun -u python predict_ddp.py --seed 2 --batch_size_val 8 --epochs 10 --dev_filepath $dev_filepath --test_filepath $test_filepath --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $max_length --offline_model_path $offline_model_path 


# (4) load the 650M_human_V11_PPI_model/35M_human_V11_PPI_model pytorch.bin from huggingface.
# Setting your own output filepath and resume_from_checkpoint (the download huggingface folder).
resume_from_checkpoint=../huggingface_650M_model/pytorch_model.bin
offline_model_path=../offline/
output_filepath=../checkpoint/run_plminteract/inference_PPI/
test_filepath=../cross_species_benchmarking/test/ecoli.ppi.qrels.seq.test.csv
mkdir -p $output_filepath
max_length=1603
srun -u python inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length $max_length --offline_model_path $offline_model_path



# (5) mutation effect training
resume_from_checkpoint=../huggingface_mutation_model/pytorch_model.bin
output_path=../checkpoint/run_plminteract/mutation_train/
mkdir -p $output_path
offline_model_path=../offline/
task_name=taskname
maxlength=2196
input_data=../mutation_data/
srun -u python mutation_train.py --epochs 50 --seed 2 --task_name $task_name --batch_size_train 1 --batch_size_val 4 --train_filepath ${input_data}train_ppi_seqs.csv --dev_filepath ${input_data}val_ppi_seqs.csv  --output_path $output_path --warmup_steps 2000 --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $maxlength --gradient_accumulation_steps 32 --offline_model_path $offline_model_path


# (6) Mutation effects prediction
model_path=../huggingface_mutation_model/
resume_from_checkpoint=${model_path}pytorch_model.bin
output=${model_path}predict/
mkdir -p $output
offline_model_path=../offline/
maxlength=3833
input_data=../mutation_data/
srun -u python mutation_predict.py --seed 2 --batch_size_val 1 --test_filepath ${input_data}decreasing_increasing_test_PPIs.csv --output_path $output --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length $maxlength --offline_model_path $offline_model_path


# (7) inference_PPI_singleGPU
resume_from_checkpoint=../huggingface_650M_model/pytorch_model.bin
test_filepath=../cross_species_benchmarking/test/ecoli.ppi.qrels.seq.test.csv
offline_model_path=../offline/
output_filepath=../checkpoint/run_plminteract/inference_PPI_singleGPU/
mkdir -p $output_filepath
max_length=1603
srun -u python inference/inference_PPI_singleGPU.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path $offline_model_path