# PLM-interact: extending protein language models to predict protein-protein interactions
<a href="https://www.biorxiv.org/content/10.1101/2024.11.05.622169v2 "><img src="https://img.shields.io/badge/Paper-bioRxiv-red" style="max-width: 100%;"></a>
<a href="https://huggingface.co/danliu1226"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-blue?label=Model" style="max-width: 100%;"></a>

Computational prediction of protein structure from amino acid sequences alone has been achieved with unprecedented accuracy, yet the prediction of protein-protein interactions (PPIs) remains an outstanding challenge. Here we assess the ability of protein language models (PLMs), routinely applied to protein folding, to be retrained for PPI prediction. Existing PPI prediction models that exploit PLMs use a pre-trained PLM feature set, ignoring that the proteins are physically interacting. Our novel method, PLM-interact, goes beyond a single protein, jointly encoding protein pairs to learn their relationships, analogous to the next-sentence prediction task from natural language processing.

![PLM-interact](https://github.com/liudan111/PLM-interact/blob/main/assets/PLM-interact.png)

## Conda env install
```
conda create -n sentence_pair python=3.10

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

git clone  https://github.com/huggingface/transformers.git
cd transformers
pip install -e .

git clone https://github.com/UKPLab/sentence-transformers.git
cd sentence-transformers
pip install -e .

pip install datasets

pip install huggingface-hub
```

## An example to predict interaction probability between proteins
```
The parameter description for this script.

(1) folder_huggingface_download : The path to a trained PLM-interact model downloaded from Hugging Face (e.g., "PLM-interact-650M-humanV11" or "PLM-interact-650M-humanV12"). Ensure the folder contains the 'pytorch_model.bin' file (2.61 GB).

(2) model_name: The Hugging Face ID for the base ESM-2 model that the PLM-interact model was built upon. Supported options are 'facebook/esm2_t33_650M_UR50D' and 'facebook/esm2_t12_35M_UR50D'.  It is critical that this base model matches the one specified in 'folder_huggingface_download'.
  - Use 'esm2_t33_650M_UR50D' for 650M-parameter models (e.g., PLM-interact-650M-humanV11, PLM-interact-650M-humanV12).
  - Use 'esm2_t12_35M_UR50D' for the 35M-parameter model (e.g., PLM-interact-35M-humanV11).

(3) embedding_size: The corresponding embedding dimension of the specified ESM-2 model. Use 1280 for 'facebook/esm2_t33_650M_UR50D' or 480 for 'facebook/esm2_t12_35M_UR50D'.

(4) max_length: The maximum sequence length for the model. This value should be the combined length of the input paired protein plus three (special tokens).
```

Download models from Huggingface 

```python
from huggingface_hub import snapshot_download
import os

# The ID of the repository you want to download
repo_id = "danliu1226/PLM-interact-650M-humanV12" # Or any other repo

# The local directory where you want to save the folder
local_dir = "../offline/PLM-interact-650M-humanV12"

# Create the directory if it doesn't exist
os.makedirs(local_dir, exist_ok=True)

print(f"Downloading repository '{repo_id}' to '{local_dir}'...")

# Use snapshot_download with force_download=True
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    force_download=True  # <-- ADD THIS LINE
)
print("\nDownload complete!")
print(f"All files for {repo_id} are saved in the '{local_dir}' folder.")
``` 

Run a simple test

```python
import torch
import torch.nn as nn
from transformers import AutoModel,AutoModelForMaskedLM,AutoTokenizer
import os
import torch.nn.functional as F

class PLMinteract(nn.Module):
  def __init__(self,model_name,num_labels,embedding_size): 
    super(PLMinteract,self).__init__() 
    self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name) 
    self.embedding_size=embedding_size
    self.classifier = nn.Linear(embedding_size,1) # embedding_size 
    self.num_labels=num_labels

  def forward_test(self,features):
    embedding_output = self.esm_mask.base_model(**features, return_dict=True)
    embedding=embedding_output.last_hidden_state[:,0,:] #cls token
    embedding = F.relu(embedding)
    logits = self.classifier(embedding)
    logits=logits.view(-1)
    probability = torch.sigmoid(logits)
    return  probability

folder_huggingface_download='download_huggingface_folder/'
model_name= 'facebook/esm2_t33_650M_UR50D'
embedding_size =1280

protein1 ="EGCVSNLMVCNLAYSGKLEELKESILADKSLATRTDQDSRTALHWACSAGHTEIVEFLLQLGVPVNDKDDAGWSPLHIAASAGRDEIVKALLGKGAQVNAVNQNGCTPLHYAASKNRHEIAVMLLEGGANPDAKDHYEATAMHRAAAKGNLKMIHILLYYKASTNIQDTEGNTPLHLACDEERVEEAKLLVSQGASIYIENKEEKTPLQVAKGGLGLILKRMVEG"

protein2= "MGQSQSGGHGPGGGKKDDKDKKKKYEPPVPTRVGKKKKKTKGPDAASKLPLVTPHTQCRLKLLKLERIKDYLLMEEEFIRNQEQMKPLEEKQEEERSKVDDLRGTPMSVGTLEEIIDDNHAIVSTSVGSEHYVSILSFVDKDLLEPGCSVLLNHKVHAVIGVLMDDTDPLVTVMKVEKAPQETYADIGGLDNQIQEIKESVELPLTHPEYYEEMGIKPPKGVILYGPPGTGKTLLAKAVANQTSATFLRVVGSELIQKYLGDGPKLVRELFRVAEEHAPSIVFIDEIDAIGTKRYDSNSGGEREIQRTMLELLNQLDGFDSRGDVKVIMATNRIETLDPALIRPGRIDRKIEFPLPDEKTKKRIFQIHTSRMTLADDVTLDDLIMAKDDLSGADIKAICTEAGLMALRERRMKVTNEDFKKSKENVLYKKQEGTPEGLYL"

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(model_name) 
PLMinter= PLMinteract(model_name, 1, embedding_size)
load_model = torch.load(f"{folder_huggingface_download}pytorch_model.bin")
PLMinter.load_state_dict(load_model,strict=False)

texts=[protein1, protein2]
tokenized = tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=1603)       
tokenized = tokenized.to(DEVICE)

PLMinter.eval()
PLMinter.to(DEVICE)
with torch.no_grad():
    probability = PLMinter.forward_test(tokenized)
    print(probability.item())
```

## Model checkpoints are available on 🤗 Hugging Face
#### Trained on human PPIs from (https://d-script.readthedocs.io/en/stable/data.html)
[danliu1226/PLM-interact-650M-humanV11](https://huggingface.co/danliu1226/PLM-interact-650M-humanV11/tree/main)

[danliu1226/PLM-interact-35M-humanV11](https://huggingface.co/danliu1226/PLM-interact-35M-humanV11/tree/main)

#### Trained on virus-human PPIs from (http://kurata35.bio.kyutech.ac.jp/LSTM-PHV/download_page)
[danliu1226/PLM-interact-650M-VH](https://huggingface.co/danliu1226/PLM-interact-650M-VH/tree/main)

#### Trained on Human PPIs from STRING V12
[danliu1226/PLM-interact-650M-humanV12](https://huggingface.co/danliu1226/PLM-interact-650M-humanV12/tree/main)


# Inference, train and evaluation 
We provide a setup script to run PLM-interact for training, validation and test. It can be found at the following path: PLM-interact/PLM-interact/script/slurm.sh. We list the commands for inference, triaing and evalaution. Please read the [PLM-interact/PLM-interact/script/README.md](PLM-interact/script/README.md) for details on parameter descriptions.

## PPI inference with multi-GPUs
```
srun -u python inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path $offline_model_path
```
* Example : 
srun -u python inference_PPI.py --seed 2 --batch_size_val 16 --test_filepath '../ppi_seq.csv' --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath '../output/' --resume_from_checkpoint '../PLM-interact-650M-humanV12/pytorch_model.bin' --max_length 1603 --offline_model_path '../dowload_path_ESM2/'


## PPI inference with a single GPU
```
srun -u python inference_PPI_singleGPU.py --seed 2 --batch_size_val 16 --test_filepath $test_filepath --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath $output_filepath --resume_from_checkpoint $resume_from_checkpoint --max_length 1603 --offline_model_path $offline_model_path
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

## Acknowledgements
Thanks to the following open-source projects:
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- [esm](https://github.com/facebookresearch/esm)
- [transformers](https://github.com/huggingface/transformers)

<img src="https://github.com/liudan111/PLM-interact/blob/main/assets/Logo.png" width="550" />
