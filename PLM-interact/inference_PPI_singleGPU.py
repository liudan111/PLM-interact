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

from typing import Dict, Type, Callable, List
import transformers
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm, trange
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
  
''' The follwing code is modified based on the CrossEncoder function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py'''
class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, tokenizer_args:Dict = {}, automodel_args:Dict = {}, default_activation_function = None, embedding_size:int=None,weight_loss_class:int=0,weight_loss_mlm:int=0,checkpoint :str=None):
        self.config = AutoConfig.from_pretrained(model_name)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.checkpoint=checkpoint

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.embedding_size = embedding_size
        self.max_length = max_length
        self.model = PLMinteract(model_name,self.config.num_labels, config=self.config,device=self.device,embedding_size=self.embedding_size)
        self.model = self.model.to(self.device)
    
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

        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        load_model = torch.load(f"{self.checkpoint}",map_location='cpu')
        self.model.load_state_dict(load_model, strict=False)
        self.predict(args,batch_size_val = batch_size_val,output_path= output_path)

    def predict(self, args,
               batch_size_val:int=32,
               output_path:str=None,
               ):
        self.model.eval()
        self.model.to(self.device)
        test_samples = load_test_objs(args.test_filepath)
        test_dataloader = DataLoader(test_samples, batch_size=batch_size_val,collate_fn = self.smart_batching_collate,shuffle=False)
        pred_scores = []
        for _, (features) in enumerate (test_dataloader):
                with torch.no_grad():
                    probability = self.model.forward_test(features)
                pred_scores.extend(probability)
        pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
        pd.DataFrame(pred_scores).to_csv(output_path + 'pred_scores.csv', index=None,header=None)

def main(args,argsDict):
    if args.seed is not None:
        random.seed(args.seed)

    model_path = args.offline_model_path+ args.model_name
    output_path=args.output_filepath
    
    trainer = CrossEncoder(model_path, num_labels=1, max_length=args.max_length, embedding_size=args.embedding_size, checkpoint = args.resume_from_checkpoint)

    trainer.inference(args,
            batch_size_val=args.batch_size_val,
            output_path= output_path,
            )

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="If the training should continue from a checkpoint folder.")
    parser.add_argument('--offline_model_path', type=str, help='offline model path')
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--data', type=str, help='data')
    parser.add_argument('--batch_size_val', default=32, type=int, help='Input train batch size on each device (default: 32)')
    parser.add_argument('--test_filepath', type=str, help='test_filepath')
    parser.add_argument('--output_filepath', type=str, help='output_filepath')
    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--embedding_size', type=int, help='embedding_size')
    parser.add_argument('--max_length', type=int, help='max_length')
 
    args = parser.parse_args()
    argsDict= args.__dict__
    main(args,argsDict)