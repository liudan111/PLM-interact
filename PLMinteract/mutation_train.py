"""
Predict mutant effects in human PPIs.
"""
import os
from torch.utils.data import WeightedRandomSampler,DataLoader,SubsetRandomSampler
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample
from datetime import datetime
import gzip
import argparse
import tarfile
import torch.distributed as dist
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import random
import math
import sys
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding,default_data_collator,AutoTokenizer, AutoConfig
import numpy as np
import logging
from typing import Dict, Type, Callable, List,NamedTuple
import transformers
import pickle
from torch import nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, util
import csv
from torch.utils.data import RandomSampler
from transformers import AutoModel,AutoModelForMaskedLM
import torch.nn.functional as F
from datetime import timedelta
import pandas as pd
from datasets import Dataset
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    auc,
    confusion_matrix,
    roc_curve,
)
from torch.optim import Optimizer
from torch.utils.data import TensorDataset
from transformers import DataCollatorForLanguageModeling
logger = logging.getLogger(__name__)
from utils.ddp import ddp_setup, distributed_concat,SequentialDistributedSampler


class ESM2_MLM_classification(nn.Module):
    def __init__(self,checkpoint,num_labels,config,device,embedding_size,weight_loss_class,weight_loss_mlm): 
        super().__init__() 
        self.esm_mask = AutoModelForMaskedLM.from_pretrained(checkpoint,config=config) 
        self.embedding_size=embedding_size
        self.classifier = nn.Linear(embedding_size,1) 
        self.num_labels=num_labels
        self.device=device
        self.weight_loss_class=weight_loss_class
        self.weight_loss_mlm=weight_loss_mlm

    def forward(self, label, lm_features):
        lm_features = lm_features.to(self.device) 
        features ={'input_ids':lm_features['input_ids'],'attention_mask':lm_features['attention_mask']}
        embedding_output = self.esm_mask.base_model(**features, return_dict=True)
        embedding=embedding_output.last_hidden_state[:,0,:] #cls token
        embedding = F.relu(embedding)
        logits = self.classifier(embedding)
        logits=logits.view(-1)
        pos_weight = torch.tensor([10], device=self.device)
        loss_fct = nn.BCEWithLogitsLoss(pos_weight= pos_weight) if self.num_labels == 1 else nn.CrossEntropyLoss()
        loss = loss_fct(logits, label.view(-1))
        return  loss,logits


class PLM_mutation_classification(nn.Module):
    def __init__(self, plm_model, device):  
        super().__init__()
        self.PLM = plm_model
        self.device = device

    def forward(self, labels, wild_features, mutant_features):
        loss_wild,  wild_logits = self.PLM.forward(labels, wild_features)
        loss_mutated, mutant_logits = self.PLM.forward(labels, mutant_features)
        logit_ratio = mutant_logits - wild_logits
        pos_weight = torch.tensor([4.5], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
        mutation_binary_loss = loss_fn(logit_ratio, labels.view(-1))
        return  mutation_binary_loss, logit_ratio

    def forward_test(self, labels, wild_features, mutant_features):
        loss_wild,  wild_logits = self.PLM.forward(labels, wild_features)
        loss_mutated,  mutant_logits = self.PLM.forward(labels, mutant_features)
        logit_ratio = mutant_logits - wild_logits
        pos_weight = torch.tensor([4.5], device=self.device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        mutation_binary_loss = loss_fn(logit_ratio, labels.view(-1))
        return  mutation_binary_loss,  logit_ratio,wild_logits,mutant_logits

''' The CrossEncoder function is modified based on the CrossEncoder function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py
and the Sentence-Transformers library is under Apache License 2.0
'''

class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None,  tokenizer_args:Dict = {}, automodel_args:Dict = {}, default_activation_function = None, embedding_size:int=None,checkpoint :str=None, weight_loss_class:int=0,weight_loss_mlm:int=0):
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

        self.weight_loss_class=weight_loss_class
        self.weight_loss_mlm=weight_loss_mlm

        self.PLM =  ESM2_MLM_classification(model_name,self.config.num_labels, config=self.config,device=self.device,embedding_size=self.embedding_size,weight_loss_class=self.weight_loss_class,weight_loss_mlm=self.weight_loss_mlm)

        self.model = PLM_mutation_classification(self.PLM, device=self.device)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)

        self.tokenizer.pad_token = self.tokenizer.eos_token
     
     
    def smart_batching_collate(self, batch):
        texts_group1 = []
        texts_group2 = []
        labels = []
        for example in batch:
            texts_group1.append(example['texts'][0])  # First text pair
            texts_group2.append(example['texts'][1])  # Second text pair
            labels.append(example['label'])  # Collect label

        tokenized_group1 = self.tokenizer(texts_group1, padding='max_length', truncation=True, return_tensors="pt",  max_length=self.max_length, pad_to_max_length=True )
        tokenized_group2 = self.tokenizer(texts_group2,  padding='max_length', truncation=True, return_tensors="pt", max_length=self.max_length,pad_to_max_length=True)

        labels = torch.tensor(labels, dtype=torch.float).to(self.device)
        for name in tokenized_group1:
            tokenized_group1[name] = tokenized_group1[name].to(self.device)
        for name in tokenized_group2:
            tokenized_group2[name] = tokenized_group2[name].to(self.device)
        return tokenized_group1, tokenized_group2, labels

    def input_example_to_dict(self, input_example):
        return {
            'texts': input_example.texts,  # list of texts for each example
            'label': input_example.label   # label (integer or float)
        }
    
    def train(self,args,
            train_samples:List[InputExample]= None,
            dev_samples:List[InputExample]= None,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = nn.Identity(),
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW, 
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            gradient_accumulation_steps: int=1,
            ):
        if use_amp:
            from torch import amp
            scaler=torch.amp.GradScaler("cuda")

        self.model = self.model.to(self.device)
        if output_path is not None:
            os.makedirs(output_path, exist_ok=True)

        train_dicts = [self.input_example_to_dict(example) for example in train_samples]
        train_dataset = Dataset.from_dict({
            'texts': [item['texts'] for item in train_dicts],
            'label': [item['label'] for item in train_dicts]
        })

        dev_dicts = [self.input_example_to_dict(example) for example in dev_samples]
        dev_dataset = Dataset.from_dict({
                        'texts': [item['texts'] for item in dev_dicts],
                        'label': [item['label'] for item in dev_dicts]
        })
        train_dataloader = DataLoader(train_dataset,batch_size=args.batch_size_train,shuffle=False,sampler =  DistributedSampler(train_dataset))
        train_dataloader.collate_fn = self.smart_batching_collate

        num_train_steps = int(len(train_dataloader) * epochs)
        param_optimizer = list(self.model.module.PLM.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        completed_steps = 1
        training_steps = 0
        starting_epoch = 0
        self.best_score = -9999999
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                load_checkpoint= torch.load(checkpoint_path,map_location='cpu')
                optimizer.load_state_dict(load_checkpoint["optimizer"])
                self.model.module.PLM.load_state_dict(load_checkpoint['model'])

        skip_scheduler = False
        for epoch in range(args.epochs):
                self.model.zero_grad()
                self.model.train()
                train_dataloader.sampler.set_epoch(epoch)
                for batch_idx, (features_wild, feature_mutant, labels) in enumerate(train_dataloader):
                    features_wild  =  features_wild.to(self.device)
                    feature_mutant  =  feature_mutant.to(self.device)
                    labels=labels.to(self.device) 

                    if use_amp:
                        if batch_idx % gradient_accumulation_steps != 0:
                            with self.model.no_sync():
                                with torch.amp.autocast("cuda"):
                                    loss_value,  logit_ratio = self.model.forward(labels,features_wild,feature_mutant)
                                    loss_value = loss_value / gradient_accumulation_steps
                                scaler.scale(loss_value).backward()
                        else:
                            with torch.amp.autocast("cuda"):
                                loss_value,  logit_ratio = self.model.forward(labels,features_wild,feature_mutant )
                                loss_value = loss_value / gradient_accumulation_steps
                            
                            scaler.scale(loss_value).backward()
                            # Gradient clipping.
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                            # weights update, Take gradient step,
                            scaler.step(optimizer)
                            scaler.update()
                            # skip scheduler or not
                            scale_before_step = scaler.get_scale()
                            skip_scheduler = scaler.get_scale() != scale_before_step
                            # Flush gradients. 
                            optimizer.zero_grad()
                            # update the learning rate of the optimizer based on the current epoch 
                            if not skip_scheduler:
                                scheduler.step()   
           
                    training_steps = training_steps+1   
                    if self.master_process: 
                        csv_headers = ['epoch','training_steps',"train_loss", 'mutation_binary_loss']
                        csv_path = os.path.join(output_path, args.task_name + '_train_loss.csv')
                        output_file_exists = os.path.isfile(csv_path)
                        with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not output_file_exists:
                                writer.writerow(csv_headers)
                            writer.writerow([epoch,training_steps,loss_value.item()])

                if self.master_process: 
                    raw_model  = self.model.module
                    checkpoint = {'model':raw_model.state_dict(),'epoch':epoch,  'loss':loss_value,'best_score': self.best_score}
                    torch.save(checkpoint, os.path.join(output_path, args.task_name + '_epoch_' + str(epoch) +'.pt'))

                    self.val(args, dev_dataset, batch_size_val= args.batch_size_val, convert_to_numpy=True,  apply_softmax= True, output_path=output_path, epoch=epoch)

    def val(self, args,dev_dataset,batch_size_val:int=32,apply_softmax = False,convert_to_numpy: bool = True,convert_to_tensor: bool = False,output_path:str=None, epoch:int=0):
                    self.model.eval()
                    self.model.to(self.device)
                    pred_scores = []
                    labels_val=[]
                    loss_value=[]

                    dev_sampler = SequentialDistributedSampler(dev_dataset)
                    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size_val,sampler= dev_sampler,shuffle=False, collate_fn = self.smart_batching_collate)

                    with torch.no_grad():
                        for batch_idx, (features_wild, feature_mutant, labels) in enumerate(dev_dataloader):
                                features_wild  =  features_wild.to(self.device)
                                feature_mutant  =  feature_mutant.to(self.device)
                                labels=labels.to(self.device) 
                                loss, logits,_, _ = self.model.module.forward_test(labels,features_wild,feature_mutant)

                                loss_value.append(loss)
                                pred_scores.extend(logits)
                                labels_val.extend(labels) 
                    pred_scores = distributed_concat(torch.stack(pred_scores),  len(dev_sampler.dataset))
                    labels_val = distributed_concat(torch.stack(labels_val),  len(dev_sampler.dataset))
                    loss_value = distributed_concat(torch.stack(loss_value),  len(dev_sampler.dataset))

                    if self.master_process:
                        pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
                        labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])
                        loss = torch.mean(loss_value)

                        pred_scores = torch.tensor(pred_scores, dtype=torch.float32)
                        pred_scores = torch.sigmoid(pred_scores)
                        ap_score = average_precision_score(labels_val, pred_scores)
                        auroc = roc_auc_score(labels_val, pred_scores)
                        logger.info(f" val_epoch:{epoch} ap_score: {ap_score} auroc: {auroc}")
                      
                        if  ap_score > self.best_score:
                                self.best_score = ap_score
                                self.best_epoch=epoch

                        csv_headers = ['best_epoch', 'epoch', "loss", "precision-recall auc","auroc"]
                        csv_path = os.path.join(output_path, args.task_name+ '_predict_metrics_val.csv')
                        output_file_exists = os.path.isfile(csv_path)
                        with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if not output_file_exists:
                                writer.writerow(csv_headers)
                            writer.writerow([self.best_epoch, epoch, loss.item(),ap_score,auroc])  

def ddp_setup():
    if 'SLURM_PROCID' in os.environ:
        os.environ['RANK'] = os.environ['SLURM_PROCID']
        rank  = int(os.environ['RANK'])
        gpus_per_node = int(os.environ['SLURM_GPUS_ON_NODE'])
        local_rank= rank - gpus_per_node * (rank // gpus_per_node)
        os.environ['LOCAL_RANK'] = str(local_rank)
    else:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
    ddp_rank       = int(os.environ['RANK'])        # Global rank for DDP.
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # Local rank for DDP.
    device         = f"cuda:{ddp_local_rank}"
    seed_offset = ddp_rank 
    return seed_offset,ddp_rank,ddp_local_rank,device
def load_train_objs(train_filepath):
    logger.info("Reading train dataset")
    train_samples = []
    with open(train_filepath, 'r', encoding='utf8') as fIn:
        # reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_ALL)
        
        for row in reader:
            train_samples.append(InputExample(
                texts=[[row['wild_seq'], row['participant_sequence']], [row['mutant_seq'], row['participant_sequence']]],
                label=int(row['label'])
            ))
            train_samples.append(InputExample(
                texts=[[row['participant_sequence'], row['wild_seq']], [row['mutant_seq'], row['participant_sequence']]],
                label=int(row['label'])
            ))
            train_samples.append(InputExample(
                texts=[[row['wild_seq'], row['participant_sequence']], [row['participant_sequence'], row['mutant_seq']]],
                label=int(row['label'])
            ))
            train_samples.append(InputExample(
                texts=[[row['participant_sequence'], row['wild_seq']], [row['participant_sequence'], row['mutant_seq']]],
                label=int(row['label'])
            ))
    return train_samples

def load_test_objs(filepath):
    logger.info("Reading test dataset")
    test_samples = []
    with open(filepath, 'r', encoding='utf8') as fIn:
        reader = csv.DictReader(fIn, delimiter=',', quoting=csv.QUOTE_NONE)
        for row in reader:
         
            test_samples.append(InputExample(
                texts=[[row['wild_seq'], row['participant_sequence']], [row['mutant_seq'], row['participant_sequence']]],
                label=int(row['label'])
            ))
    return test_samples

class MutationTrainArguments(NamedTuple):
    epochs:int
    offline_model_path:str
    resume_from_checkpoint:str
    seed:int
    task_name:str
    batch_size_train:int
    batch_size_val:int
    train_filepath:str
    dev_filepath:str
    output_path:str
    model_name:str
    embedding_size:int
    warmup_steps:int
    gradient_accumulation_steps:int
    max_length:int
    weight_loss_mlm:int
    weight_loss_class:int
    func: Callable[["MutationTrainArguments"], None]

def add_args_func(parser):
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--offline_model_path', type=str, help='offline model path')
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="If the training should continue from a checkpoint folder.")
    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--task_name', type=str, help='task_name')
    parser.add_argument('--batch_size_train', default=16, type=int, help='Input train batch size on each device (default: 16)')
    parser.add_argument('--batch_size_val', default=16, type=int, help='Input val batch size on each device (default: 16)')
    parser.add_argument('--train_filepath', type=str, help='train_filepath')
    parser.add_argument('--dev_filepath', type=str, help='dev_filepath')
    parser.add_argument('--output_path', type=str, help='output_path')
    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--embedding_size', type=int, help='embedding_size')
    parser.add_argument('--warmup_steps', default=2000,type=int, help='warmup_steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, help='max_length')
    parser.add_argument('--weight_loss_mlm', default=1, type=int, help='weight_loss_mlm')  
    parser.add_argument('--weight_loss_class', default=1, type=int, help='weight_loss_class') 
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
    offline = args.offline_model_path
    model_path = offline+ args.model_name

    resume_from_checkpoint= args.resume_from_checkpoint
    output_path= args.output_path

    trainer = CrossEncoder(model_path, num_labels=1, max_length=args.max_length, embedding_size=args.embedding_size,checkpoint = resume_from_checkpoint, weight_loss_class=args.weight_loss_class,weight_loss_mlm=args.weight_loss_mlm)

    train_samples, dev_samples = load_train_objs(args.train_filepath), load_test_objs(args.dev_filepath)
    
    trainer.train(args,
                train_samples= train_samples,
                dev_samples= dev_samples,
                epochs = args.epochs,
                warmup_steps= args.warmup_steps,
                output_path= output_path,
                use_amp=True,
                gradient_accumulation_steps=args.gradient_accumulation_steps
                )
    destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    add_args_func(parser)
    args = parser.parse_args()
    main(args)