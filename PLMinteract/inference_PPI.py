"""
PPI prediction.
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

from typing import Dict, Type, Callable, List,NamedTuple
import transformers
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

from utils.data_load import load_test_objs
from utils.ddp import ddp_setup, distributed_concat,SequentialDistributedSampler

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
logger = logging.getLogger(__name__)

class PLMinteract(nn.Module):
  def __init__(self,model_name,num_labels,config,device,embedding_size): 
    super(PLMinteract,self).__init__() 
    self.esm_mask = AutoModelForMaskedLM.from_pretrained(model_name,config=config) 
    self.embedding_size=embedding_size
    self.classifier = nn.Linear(embedding_size,1) # embedding_size 
    self.num_labels=num_labels
    self.device=device

  def forward_test(self,features):
    embedding_output = self.esm_mask.base_model(**features, return_dict=True)
    embedding=embedding_output.last_hidden_state[:,0,:] #cls token
    embedding = F.relu(embedding)
    logits = self.classifier(embedding)
    logits=logits.view(-1)
    probability = torch.sigmoid(logits)
    return  probability
  
''' The CrossEncoder function is modified based on the CrossEncoder function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py
and the Sentence-Transformers library is under Apache License 2.0
'''
class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, tokenizer_args:Dict = {}, automodel_args:Dict = {}, default_activation_function = None, embedding_size:int=None,weight_loss_class:int=0,weight_loss_mlm:int=0,checkpoint :str=None):
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

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.model = PLMinteract(model_name,self.config.num_labels, config=self.config,device=self.device,embedding_size=self.embedding_size)

        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)

    def smart_batching_collate(self,batch):
            texts = [[] for _ in range(len(batch[0].texts))]
            for example in batch:
                for idx, text in enumerate(example.texts):
                    texts[idx].append(text.strip())
            tokenized = self.tokenizer(*texts, padding=True, truncation='longest_first', return_tensors="pt", max_length=self.max_length)
          
            for name in tokenized:
                tokenized[name] = tokenized[name].to(self.device)
            return tokenized
  
    def inference(self,args,
            batch_size_val: int = 1,        
            output_path: str = None,
            show_progress_bar: bool = True,
            ):
        self.model = self.model.to(self.device)

        if output_path is not None and self.master_process:
            os.makedirs(output_path, exist_ok=True)

        load_model = torch.load(f"{self.checkpoint}",map_location='cpu')
        self.model.module.load_state_dict(load_model,strict=False)

        self.predict(args,batch_size_val = batch_size_val,output_path= output_path)

    def predict(self, args,
               batch_size_val:int=32,
               output_path:str=None,
               ):
        self.model.eval()
        self.model.to(self.device)
        test_samples = load_test_objs(args.test_filepath)
        test_sampler = SequentialDistributedSampler(test_samples)
        test_dataloader = DataLoader(test_samples, batch_size=batch_size_val,sampler= test_sampler,collate_fn = self.smart_batching_collate,shuffle=False)

        pred_scores = []
        for _, (features) in enumerate (test_dataloader):
                with torch.no_grad():
                    probability = self.model.module.forward_test(features)
                pred_scores.extend(probability)
        pred_scores = distributed_concat(torch.stack(pred_scores), len(test_sampler.dataset))
      
        if self.master_process:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
            pd.DataFrame(pred_scores).to_csv(output_path + 'pred_scores.csv', index=None,header=None)

class InferenceArguments(NamedTuple):
    resume_from_checkpoint:str
    offline_model_path:str
    seed:int
    batch_size_val:int
    test_filepath:str
    output_filepath:str
    model_name:str
    embedding_size:int
    max_length:int
    func: Callable[["InferenceArguments"], None]

def add_args_func(parser):
    parser.add_argument('--seed', type=int, default=2, help='Random seed for reproducibility (default: 2).')
    
    parser.add_argument('--test_filepath', type=str, required=True, help='Path to the test dataset (CSV format).')
    parser.add_argument('--output_filepath', type=str, required=True, help='Path to save the prediction results.')

    parser.add_argument("--resume_from_checkpoint",type=str,required=True, help="Path to a trained model (default: None).")
    parser.add_argument('--batch_size_val', default=16, type=int, help='The validation batch size on each device (default: 16).')
    parser.add_argument('--max_length', type=int, default=1603, help='Maximum sequence length for tokenizing paired proteins (default: 1603).')

    parser.add_argument('--offline_model_path', type=str, required=True, help='Path to a locally stored ESM-2 model.')
    parser.add_argument('--model_name', type=str, required=True, help='Choose the ESM-2 model to load (esm2_t12_35M_UR50D / esm2_t33_650M_UR50D).')
    parser.add_argument('--embedding_size', type=int, required=True, help='Set embedding vector size based on the selected ESM-2 model (480 / 1280).')
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

    model_path = args.offline_model_path+ args.model_name
    output_path=args.output_filepath
    
    trainer = CrossEncoder(model_path, num_labels=1, max_length=args.max_length, embedding_size=args.embedding_size, checkpoint = args.resume_from_checkpoint)
    trainer.inference(args,
            batch_size_val=args.batch_size_val,
            output_path= output_path,
            )
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args_func(parser)
    args = parser.parse_args()
    main(args)