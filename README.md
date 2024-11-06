# PLM-interact
PLM-interact: extending protein language models to predict protein-protein interactions.



![Image Alt Text](https://github.com/liudan111/PLM-interact/blob/main/assets/PLM-interact.tiff)

## PPI Prediction



## Train
The efficent batch size is 128, which is equal to  batch_size_train * gradient_accumulation_steps * the number of gpus

### PLM-interact training with mask loss and binary classification loss optimize
srun -u python cls_class_train.py --epochs 10 --seed 2 --data 'human_V11' --task_name '1vs10' --batch_size_train 1 --batch_size_val 16 --train_filepath 'pairs_seqs_train_human_virus_len_2146.tsv' --dev_filepath 'pairs_seqs_val_human_virus_len_2154.tsv' --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --output_filepath outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 8 --max_length 2146 --weight_loss_mlm 1 --weight_loss_class 10

### PLM-interact training with binary classification loss optimize
srun -u python train_binary.py --epochs 10 --seed 2 --data 'human_V11' --task_name 'binary' --batch_size_train 1 --batch_size_val 32 --train_filepath pairs_seqs_human_virus_train.tsv --dev_filepath pairs_seqs_human_virus_val.tsv --test_filepath pairs_seqs_human_virus_test.tsv --output_filepath outputfilepath --warmup_steps 2000 --gradient_accumulation_steps 32  --model_name 'esm2_t33_650M_UR50D' --embedding_size 1280 --max_length 1600 --evaluation_steps 5000 --sub_samples 5000


## Model checkpoints
[danliu1226/PLM-interact-650M-humanV11](https://huggingface.co/danliu1226/PLM-interact-650M-humanV11/upload/main)


## Acknowledgements

Thanks to the following open-source projects:
- [sentence_transformers](https://github.com/UKPLab/sentence-transformers)
- [esm](https://github.com/facebookresearch/esm)
- [transformers](https://github.com/huggingface/transformers)

The authors acknowledge funding from the European Union’s Horizon 2020 research and innovation 562 program, under the Marie Sklodowska-Curie Actions Innovative Training Networks 563 grant agreement no. 955974 (VIROINF) for DL, a UK Medical Research Council (MRC) Doctoral Training Programme in Precision Medicine studentship (MR/N013166/1) for KDL and MRC grants: MC_UU_00034/5, MC_UU_00034/6 and MR/V01157X/1. This work used the DiRAC Extreme Scaling service (Tursa) at the University of Edinburgh, managed by the Edinburgh Parallel Computing Centre on behalf of the STFC DiRAC HPC Facility (www.dirac.ac.uk). The DiRAC service at Edinburgh was funded by BEIS, UKRI and STFC capital funding and STFC operations grants. DiRAC is part of the UKRI Digital Research Infrastructure. 