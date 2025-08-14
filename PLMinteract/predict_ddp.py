"""
Choose the best trained checkpoints by testing on the validation datasets and evaluate the model's performance on the test datasets.
"""
import os
from datasets import Dataset
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample
import logging
from datetime import datetime
import gzip
import numpy as np
import pandas as pd
import argparse

from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import random
import math
import sys
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

import logging
import os
from typing import Dict, Type, Callable, List,NamedTuple
import transformers
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
from sentence_transformers.evaluation import SentenceEvaluator
import csv
from torch.utils.data import RandomSampler

from transformers import AutoModel,AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from transformers.utils import check_min_version, send_example_telemetry

from transformers.utils.versions import require_version
from datetime import timedelta

from utils.data_load import load_train_objs, load_val_objs
from utils.ddp import ddp_setup, distributed_concat,SequentialDistributedSampler
from utils.metrics import find_best_acc_and_threshold,find_best_f1_and_threshold

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

class PLMinteract(nn.Module):
  def __init__(self,checkpoint,num_labels,config,device,embedding_size): 
    super(PLMinteract,self).__init__() 
    self.esm_mask = AutoModelForMaskedLM.from_pretrained(checkpoint,config=config) 
    self.embedding_size=embedding_size
    self.classifier = nn.Linear(embedding_size,1) # embedding_size 
    self.num_labels=num_labels
    self.device=device

  def forward_test(self, label,features):
    embedding_output = self.esm_mask.base_model(**features, return_dict=True)
    embedding=embedding_output.last_hidden_state[:,0,:] #cls token
    embedding = F.relu(embedding)
    logits = self.classifier(embedding)
    logits=logits.view(-1)

    pos_weight = torch.tensor([10]).to(self.device)
    loss_fct = nn.BCEWithLogitsLoss(pos_weight= pos_weight) if self.num_labels == 1 else nn.CrossEntropyLoss()
    loss = loss_fct(logits, label.view(-1))
    probability = torch.sigmoid(logits)
 
    return  loss, logits,probability
  
''' The CrossEncoder function is modified based on the CrossEncoder function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py
and the Sentence-Transformers library is under Apache License 2.0
'''

class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, tokenizer_args:Dict = {}, automodel_args:Dict = {}, default_activation_function = None, embedding_size:int=None,checkpoint :str=None):
        self.config = AutoConfig.from_pretrained(model_name)
        if 'SLURM_PROCID' in os.environ:
            os.environ['RANK'] = os.environ['SLURM_PROCID']
            self.rank  = int(os.environ['RANK'])
            gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
            local_rank= self.rank - gpus_per_node * (self.rank // gpus_per_node)
            os.environ['LOCAL_RANK'] = str(local_rank)
            self.local_rank = int(os.environ['LOCAL_RANK'])
            self.device = torch.device("cuda", local_rank)
            self.master_process = self.rank == 0  # Main process does logging & checkpoints.
            self.num_processes = torch.distributed.get_world_size()
            self.checkpoint=checkpoint
        else:
            self.local_rank =  int(os.environ['LOCAL_RANK'])
            self.rank = int(os.environ["RANK"])
            self.device = torch.device("cuda", self.local_rank)
            self.master_process = self.rank == 0  # Main process does logging & checkpoints.
            self.num_processes = torch.distributed.get_world_size()
            self.checkpoint=checkpoint

        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels
   
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.embedding_size = embedding_size
        self.max_length = max_length

        self.model = PLMinteract(model_name,self.config.num_labels, config=self.config,device=self.device,embedding_size=self.embedding_size)

        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)
  
    def smart_batching_collate(self,batch):
            texts = [[] for _ in range(len(batch[0].texts))]
            labels = []

            for example in batch:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text.strip())
                labels.append(example.label)
            
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
            labels = torch.tensor(labels, dtype=torch.float if self.config.num_labels == 1 else torch.long).to(self.device)
        
            for name in tokenized:
                tokenized[name] = tokenized[name].to(self.device)

            return tokenized, labels
  
    def eval_predict(self,args,
            batch_size_val: int = 1,        
            loss_fct = None,
            output_path: str = None,
            show_progress_bar: bool = True,
            ):
        self.model = self.model.to(self.device)
        if output_path is not None and self.master_process:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        self.best_epoch=0

        if loss_fct is None:
            pos_weight = torch.tensor([10]).to(self.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight= pos_weight) if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        dev_samples = load_val_objs(args.dev_filepath)
        for epoch in range(args.epochs):
                checkpoint_path = args.resume_from_checkpoint + 'epoch_' + str(epoch) +'.pt'
                load_checkpoint = torch.load(checkpoint_path, map_location='cpu')
                self.model.module.load_state_dict(load_checkpoint['model'])
                self.eval(args,dev_samples=dev_samples,epoch=epoch,batch_size_val = batch_size_val, output_path= output_path,loss_fct=loss_fct)

        checkpoint_path = args.resume_from_checkpoint + 'epoch_' + str(self.best_epoch) +'.pt'
        load_checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.module.load_state_dict(load_checkpoint['model'])
        self.predict(args,batch_size_val = batch_size_val,output_path= output_path,loss_fct=loss_fct)

    def eval(self,args,
            dev_samples:List[InputExample],
            epoch:int=0,
            batch_size_val: int = 32,  
            output_path: str = None,
            loss_fct=None
            ):

            dev_sampler = SequentialDistributedSampler(dev_samples)
            dev_dataloader = DataLoader(dev_samples, batch_size=batch_size_val,sampler =  dev_sampler,collate_fn = self.smart_batching_collate,shuffle=False)

            self.model.eval()
            self.model.to(self.device)
            pred_scores = []
            labels_val=[]
            loss_value=[]
            for _, (features,labels) in enumerate (dev_dataloader):
                    with torch.no_grad():
                        loss, logits,probability = self.model.module.forward_test(labels,features)
                        loss_value.append(loss)
                    labels_val.extend(labels) 
                    pred_scores.extend(logits)
                    
            pred_scores = distributed_concat(torch.stack(pred_scores),   len(dev_sampler.dataset))
            labels_val = distributed_concat(torch.stack(labels_val),  len(dev_sampler.dataset))
            loss_value = distributed_concat(torch.stack(loss_value),  len(dev_sampler.dataset))

            if self.master_process:
                pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
                labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])
                loss = torch.mean(loss_value)
                ap_score = average_precision_score(labels_val, pred_scores)

                if(ap_score>self.best_score):
                        self.best_score=ap_score
                        self.best_epoch= epoch

                csv_headers = ['epoch', "val_loss", "precision-recall auc"]
                csv_path = os.path.join(output_path + 'predict_metrics_val.csv')
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(csv_headers)
                    writer.writerow([epoch,loss.item(),ap_score])  
    
    def predict(self, args,
               batch_size_val:int=32,
               output_path:str=None,
               loss_fct=None,
               ):
        self.model.eval()
        self.model.to(self.device)
        test_samples = load_val_objs(args.test_filepath)
        test_sampler = SequentialDistributedSampler(test_samples)
        test_dataloader = DataLoader(test_samples, batch_size=args.batch_size_val,sampler= test_sampler,collate_fn = self.smart_batching_collate,shuffle=False)

        pred_scores = []
        labels_val=[]
        loss_value=[]
        for _, (features,labels) in enumerate (test_dataloader):
                with torch.no_grad():
                    loss, logits, probability = self.model.module.forward_test(labels,features)
                    loss_value.append(loss)
                pred_scores.extend(probability)
                labels_val.extend(labels) 

        pred_scores = distributed_concat(torch.stack(pred_scores),  len(test_sampler.dataset))
        labels_val = distributed_concat(torch.stack(labels_val),  len(test_sampler.dataset))
        loss_value = distributed_concat(torch.stack(loss_value),  len(test_sampler.dataset))

        if self.master_process:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
            labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])
            loss = torch.mean(loss_value)
    
            ap_score = average_precision_score(labels_val, pred_scores)
            max_acc, best_threshold =  find_best_acc_and_threshold(pred_scores,labels_val)
            f1, precision, recall, f1_threshold = find_best_f1_and_threshold(pred_scores,labels_val)

            pd.DataFrame(pred_scores).to_csv(output_path + 'pred_scores.csv', index=None, header=None)
            csv_headers = ['epoch',"test_loss", "precision-recall auc", 'f1', 'precision', 'recall', 'f1_threshold',  'max_acc', 'best_threshold']
            csv_path = os.path.join(output_path + 'predict_metrics_test.csv')
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(csv_headers)
                writer.writerow([self.best_epoch, loss.item(), ap_score,f1, precision, recall, f1_threshold, max_acc, best_threshold])  

class PredictionArguments(NamedTuple):
    epochs:int
    resume_from_checkpoint:str
    offline_model_path:str
    seed:int
    batch_size_val:int
    test_filepath:str
    dev_filepath:str
    output_filepath:str
    model_name:str
    embedding_size:int
    max_length:int
    func: Callable[["PredictionArguments"], None]

def add_args_func(parser):
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility (default: 2).')
    data_parser = parser.add_argument_group("Input data and output results")
    plminteract_parser = parser.add_argument_group("PLM-interact setting")
    ESM2_parser = parser.add_argument_group("ESM2 model loading")

    data_parser.add_argument('--dev_filepath', required=True,  type=str, help='Path to the validation dataset (CSV format).')
    data_parser.add_argument('--test_filepath', required=True,  type=str, help='Path to the test dataset (CSV format).')
    data_parser.add_argument('--output_filepath', required=True, type=str, help='Path to save validation and test results.')

    plminteract_parser.add_argument('--epochs', type=int, default=10, help='Total epochs of trained models (default: 10).' )
    plminteract_parser.add_argument("--resume_from_checkpoint",type=str,default=None,help= "Path to trained models(default: None).")
    plminteract_parser.add_argument('--batch_size_val', default=16, type=int, help='The validation batch size on each device (default: 16)')
    plminteract_parser.add_argument('--max_length', type=int, default=1603, help='Maximum sequence length for tokenizing paired proteins (default: 1603).')

    ESM2_parser.add_argument('--offline_model_path', type=str, required=True, help='Path to a locally stored ESM-2 model.')
    ESM2_parser.add_argument('--model_name', type=str, required=True, help='Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).')
    ESM2_parser.add_argument('--embedding_size', type=int, required=True, help='Set embedding vector size based on the selected ESM-2 model (480 / 1280).') 

    return parser

def main(args):
    seed_offset,ddp_rank,ddp_local_rank,device= ddp_setup()
    init_process_group(backend='nccl')
    torch.cuda.set_device(device)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior
        torch.backends.cudnn.benchmark = False  # Disable optimizations that might cause non-determinism

    logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])

    model_path = args.offline_model_path+ args.model_name
    output_path=args.output_filepath

    trainer = CrossEncoder(model_path, num_labels=1, max_length=args.max_length, embedding_size=args.embedding_size, checkpoint = args.resume_from_checkpoint)

    trainer.eval_predict(args,
            batch_size_val=args.batch_size_val,
            output_path= output_path,
            )
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args_func(parser)
    args = parser.parse_args()
    main(args)