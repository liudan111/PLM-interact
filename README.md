# PLM-interact
PLM-interact: extending protein language models to predict protein-protein interactions.

Computational prediction of protein structure from amino acid sequences alone has been achieved with unprecedented accuracy, yet the prediction of protein-protein interactions (PPIs) remains an outstanding challenge. Here we assess the ability of protein language models (PLMs), routinely applied to protein folding, to be retrained for PPI prediction. Existing PPI prediction models that exploit PLMs use a pre-trained PLM feature set, ignoring that the proteins are physically interacting. Our novel method, PLM-interact, goes beyond a single protein, jointly encoding protein pairs to learn their relationships, analogous to the next-sentence prediction task from natural language processing.

![PLM-interact](https://github.com/liudan111/PLM-interact/blob/main/assets/PLM-interact.png)

## Preprint

## Conda env install
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone  https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

git clone https://github.com/UKPLab/sentence-transformers.git
cd sentence-transformers
pip install -e .

## Model checkpoints are available on Hugging Face
Trained on human PPIs from [text](https://d-script.readthedocs.io/en/stable/data.html)
[danliu1226/PLM-interact-650M-humanV11](https://huggingface.co/danliu1226/PLM-interact-650M-humanV11/upload/main)

[danliu1226/PLM-interact-35M-humanV11](https://huggingface.co/danliu1226/PLM-interact-35M-humanV11/tree/main)

Trained on virus-human PPIs from [text](http://kurata35.bio.kyutech.ac.jp/LSTM-PHV/download_page)
[danliu1226/PLM-interact-650M-VH](https://huggingface.co/danliu1226/PLM-interact-650M-VH/tree/main)

Trained on Human PPIs from STRING V12
[danliu1226/PLM-interact-650M-humanV12](https://huggingface.co/danliu1226/PLM-interact-650M-humanV12/tree/main)



## PPI inference with multi-GPUs
srun -u python inference_ddp.py --seed 2 --data 'test' --task_name 'inference' --epochs 10 --batch_size_val 16 --dev_filepath ${train_filepath} --test_filepath ${test_filepath} --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1000 --offline_model_path $offline_model_path

## PLM-interact training
The efficent batch size is 128, which is equal to  batch_size_train * gradient_accumulation_steps * the number of gpus

### (1) PLM-interact training with mask loss and binary classification loss optimize
srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length 2146 --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 

### (2) PLM-interact training with binary classification loss optimize
srun -u python train_binary.py --epochs 20 --seed 2 --data 'human_V11' --task_name 'binary' --batch_size_train 1 --batch_size_val 32 --train_filepath $train_filepath  --dev_filepath $dev_filepath  --test_filepath $test_filepath --output_filepath $outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 32  --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --max_length 1600 --evaluation_steps 5000 --sub_samples 5000 --offline_model_path $offline_model_path 



## Acknowledgements
Thanks to the following open-source projects:
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- [esm](https://github.com/facebookresearch/esm)
- [transformers](https://github.com/huggingface/transformers)



<img src="https://github.com/liudan111/PLM-interact/blob/main/assets/logo/Logo.png" width="200" />
