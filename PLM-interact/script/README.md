# Script Modes and Input Formats

The SLURM script operates in three modes: training, evaluation, and inference. The input file must be a CSV and its required columns depend on the mode.

## Inference: 
### Required Input (--test_filepath): A CSV file with the following two columns:
    query: The sequence of the first protein.
    text: The sequence of the second protein.

## Training & Evaluation: 
### Required Input (--train_filepath,--dev_filepath, --test_filepath): A CSV file with the following three columns:
    query: The sequence of the first protein.
    text: The sequence of the second protein.
    label: The ground truth label, where 1 indicates a positive interaction and 0 indicates a negative one.

## Key Parameter description (Configuring Parameters for Your Task)

```
--train_filepath: Required. Path to the training data, which must be a CSV file. It should contain the columns: query (protein sequence 1), text (protein sequence 2), and label (1 for a positive interaction, 0 for negative).

--dev_filepath: Required. Path to the validation data, which must be a CSV file. It must follow the same format as the training file, containing query, text, and label columns.

--test_filepath: Required. Path to the test data CSV file. The required format depends on the script being run:
    * For Inference (inference_PPI.py): The CSV file must contain query (protein sequence 1) and text (protein sequence 2) columns.
    * For Training/Evaluation: The CSV file must contain query, text, and label columns.

--resume_from_checkpoint: Optional for training, Required for inference. Path to a local directory containing a fine-tuned model checkpoint (pytorch_model.bin) downloaded from the Hugging Face Hub ([danliu1226/PLM-interact-650M-humanV11](https://huggingface.co/danliu1226/PLM-interact-650M-humanV11/tree/main), [danliu1226/PLM-interact-650M-humanV12](https://huggingface.co/danliu1226/PLM-interact-650M-humanV12/tree/main)).
* Example : A folder named PLM-interact-650M-humanV11 containing the model files from danliu1226/PLM-interact-650M-humanV11.

--max_length: Required. The maximum total sequence length for a protein pair after tokenization. It should account for the combined length of paired protein plus three special tokens. To avoid truncation, it's best to set this based on the longest pair in your dataset.

--embedding_size: Required. The embedding dimension size of the base ESM-2 model. This must correspond to the model specified in offline_model_path.
    * Use 1280 for the 'esm2_t33_650M_UR50D' model.
    * Use 480 for the 'esm2_t12_35M_UR50D' model.

--offline_model_path: Required. Path to the local directory containing the base ESM-2 model downloaded from Hugging Face Hub([facebook/esm2_t33_650M_UR50D](https://huggingface.co/facebook/esm2_t33_650M_UR50D)). The model specified here must match the checkpoint provided in --resume_from_checkpoint.
    *Use 'esm2_t33_650M_UR50D' for 650M-parameter models (e.g., PLM-interact-650M-humanV11, PLM-interact-650M-humanV12).
    *Use 'esm2_t12_35M_UR50D' for the 35M-parameter model (e.g., PLM-interact-35M-humanV11).

--output_filepath: Required. Path to the directory where output files will be saved.

```

## Download trained models/checkpoints from Huggingface
```python
from transformers import AutoTokenizer, EsmModel
import torch
tokenizer = AutoTokenizer.from_pretrained("facebook/esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("facebook/esm2_t33_650M_UR50D")
tokenizer.save_pretrained("offline/esm2_t33_650M_UR50D")
model.save_pretrained("offline/esm2_t33_650M_UR50D")
```

## PPI inference with multi-GPUs

```
srun -u python inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path $offline_model_path
```

## PLM-interact training and evaluation
The efficient batch size is 128, which is equal to  batch_size_train * gradient_accumulation_steps * the number of gpus

### (1) PLM-interact training with mask loss and binary classification loss optimize
```
srun -u python train_mlm.py --epochs 20 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --train_filepath $train_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length 2146 --weight_loss_mlm 1 --weight_loss_class 10 --offline_model_path $offline_model_path 
```
### (2) PLM-interact training with binary classification loss optimize

```
srun -u python train_binary.py --epochs 20 --seed 2 --data 'human_V11' --task_name 'binary' --batch_size_train 1 --batch_size_val 32 --train_filepath $train_filepath  --dev_filepath $dev_filepath  --test_filepath $test_filepath --output_filepath $outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 32  --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --max_length 1600 --evaluation_steps 5000 --sub_samples 5000 --offline_model_path $offline_model_path 
```

### (3) PLM-interact validation and test
```
srun -u python predict_ddp.py --seed 2 --batch_size_val 32 --dev_filepath $dev_filepath --test_filepath $test_filepath --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --model_name esm2_t33_650M_UR50D --embedding_size 1280 --max_length 1603 --offline_model_path $offline_model_path 
```
