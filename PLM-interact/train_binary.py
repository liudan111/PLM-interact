import os
import datasets
from datasets import Dataset
from torch.utils.data import WeightedRandomSampler,DataLoader,SubsetRandomSampler
from sentence_transformers import LoggingHandler, util
from sentence_transformers import InputExample
from datetime import datetime
import pandas as pd
import csv
import argparse
import numpy as np
import logging
from typing import Dict, Type, Callable, List
import torch
import random
import math
import sys

from torch.utils.data import RandomSampler
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from torch import nn
from torch.optim import Optimizer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from transformers import AutoModel,AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
import torch.nn.functional as F
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from datetime import timedelta
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from utils.data_load import load_train_objs, load_val_objs
from utils.ddp import ddp_setup, distributed_concat,SequentialDistributedSampler

logger = logging.getLogger(__name__)

''' This code is modified based on the CrossEncoder function from the Sentence-Transformers library: 
https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/cross_encoder/CrossEncoder.py'''

class CrossEncoder():
    def __init__(self, model_name:str, num_labels:int = None, max_length:int = None, tokenizer_args:Dict = {}, automodel_args:Dict = {}, default_activation_function = None,checkpoint :str=None):
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
            self.checkpoint=checkpoint

        else:
            self.local_rank =  int(os.environ['LOCAL_RANK'])
            self.rank = int(os.environ["RANK"])
            self.device = torch.device("cuda", self.local_rank)
            self.master_process = self.rank == 0  # Main process does logging & checkpoints.
            self.checkpoint=checkpoint

        classifier_trained = True
        if self.config.architectures is not None:
            classifier_trained = any([arch.endswith('ForSequenceClassification') for arch in self.config.architectures])

        if num_labels is None and not classifier_trained:
            num_labels = 1

        if num_labels is not None:
            self.config.num_labels = num_labels

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config, **automodel_args)

        if(checkpoint!=None): 
            load_checkpoint= torch.load(self.checkpoint,map_location='cpu')
            self.model.load_state_dict(load_checkpoint['model'])

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_args)
        self.max_length = max_length

        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.local_rank],find_unused_parameters=True)
        if default_activation_function is not None:
            self.default_activation_function = default_activation_function
            try:
                self.config.sbert_ce_default_activation_function = util.fullname(self.default_activation_function)
            except Exception as e:
                logger.warning("Was not able to update config about the default_activation_function: {}".format(str(e)) )
        elif hasattr(self.config, 'sbert_ce_default_activation_function') and self.config.sbert_ce_default_activation_function is not None:
            self.default_activation_function = util.import_from_string(self.config.sbert_ce_default_activation_function)()
        else:
            self.default_activation_function = nn.Sigmoid() if self.config.num_labels == 1 else nn.Identity()

    def smart_batching_collate(self, batch):
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

    
    def train(self,args,
            train_dataloader: DataLoader,
            train_samples:List[InputExample]= None,
            dev_samples:List[InputExample]= None,
            batch_size_train: int = 1,
            batch_size_val: int = 1,         
            save_every_steps:int=1000,
            epochs: int = 1,
            loss_fct = None,
            activation_fct = None,
            scheduler: str = 'WarmupLinear',
            warmup_steps: int = 10000,
            optimizer_class: Type[Optimizer] = torch.optim.AdamW, 
            optimizer_params: Dict[str, object] = {'lr': 2e-5},
            weight_decay: float = 0.01,
            evaluation_steps: int = 0,
            eval_iters:int=1,
            output_path: str = None,
            save_best_model: bool = True,
            max_grad_norm: float = 1,
            use_amp: bool = False,
            callback: Callable[[float, int, int], None] = None,
            show_progress_bar: bool = True,
            gradient_accumulation_steps: int=1,
            sub_samples:int=100
            ):
     
        train_dataloader.collate_fn = self.smart_batching_collate

        if activation_fct is None:
            activation_fct = self.default_activation_function

        if use_amp:
            from torch.cuda.amp import autocast
            scaler = torch.cuda.amp.GradScaler()

        self.model = self.model.to(self.device)

        if output_path is not None and self.master_process:
            os.makedirs(output_path, exist_ok=True)

        self.best_score = -9999999
        num_train_steps = int(len(train_dataloader) * epochs)

        # Prepare optimizers
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        if isinstance(scheduler, str):
            scheduler = SentenceTransformer._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        if loss_fct is None:
            pos_weight = torch.tensor([10]).to(self.device)
            loss_fct = nn.BCEWithLogitsLoss(pos_weight= pos_weight) if self.config.num_labels == 1 else nn.CrossEntropyLoss()

        completed_steps = 1
        starting_epoch = 0
        training_steps = 0

       # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataloader)}")
        logger.info(f"  Num Epochs = {args.epochs}")
        logger.info(f"  Instantaneous batch size per device = {batch_size_train}")
        logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
                checkpoint_path = args.resume_from_checkpoint
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                optimizer.load_state_dict(checkpoint["optimizer"])
                scheduler.load_state_dict(checkpoint["scheduler"])
                starting_epoch=checkpoint['epoch']+1

        skip_scheduler = False
        for epoch in range(starting_epoch, epochs):
            self.model.zero_grad()
            self.model.train()
            train_dataloader.sampler.set_epoch(epoch)

            for batch_idx, (features, labels) in enumerate(train_dataloader):
                features=features.to(self.device)
                labels=labels.to(self.device)
                if use_amp:
                    if batch_idx % gradient_accumulation_steps != 0:
                        with self.model.no_sync():
                            with autocast():
                                model_predictions = self.model(**features, return_dict=True)
                                logits = activation_fct(model_predictions.logits)
                                if self.config.num_labels == 1:
                                    logits = logits.view(-1)
                                loss = loss_fct(logits, labels)
                                loss = loss / gradient_accumulation_steps
                            scaler.scale(loss).backward()
                    else:
                        with autocast():
                            model_predictions = self.model(**features, return_dict=True)
                            logits = activation_fct(model_predictions.logits)# Forward Pass
                            if self.config.num_labels == 1:
                                logits = logits.view(-1)
                            loss = loss_fct(logits, labels)
                            loss = loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()
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
                        optimizer.zero_grad(set_to_none=True)
                        # update the learning rate of the optimizer based on the current epoch 
                        if not skip_scheduler:
                            scheduler.step() 

                training_steps += 1
                if self.master_process and (training_steps % args.evaluation_steps == 0): 
                            csv_headers = ['epoch','training_steps',"completed_steps","train_loss",'ap_auc']
                            csv_path = os.path.join(output_path, 'train_loss_metrics.csv')
                            output_file_exists = os.path.isfile(csv_path)
                            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                                writer = csv.writer(f)
                                if not output_file_exists:
                                    writer.writerow(csv_headers)
                                writer.writerow([epoch,training_steps,completed_steps,loss.item()])
                            completed_steps += 1  
  
                if (training_steps % args.evaluation_steps == 0): 
                    self.train_eval(args,train_samples=train_samples,batch_size_val = batch_size_val,training_steps=training_steps, sub_samples=args.sub_samples, output_path= output_path,loss_fct=loss_fct)

                    self.predict(args,batch_size_val = batch_size_val,training_steps=training_steps, sub_samples = args.sub_samples, output_path= output_path,loss_fct=loss_fct)
                    
                    self.model.zero_grad()
                    self.model.train()

            if self.master_process:
                    raw_model  = self.model.module 
                    checkpoint = {'model':raw_model.state_dict(), 'optimizer':optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'epoch':epoch,'loss':loss}
                    torch.save(checkpoint, os.path.join(output_path, 'epoch_'+ str(epoch)+'.pt'))               

    def train_eval(self,args,
            train_samples:List[InputExample],
            batch_size_val: int = 32,  
            training_steps:int=1,
            sub_samples=100,
            output_path: str = None,
            activation_fct = None,
            loss_fct=None
            ):

            if activation_fct is None:
                activation_fct = self.default_activation_function

            random.seed(args.seed)
            random.shuffle(train_samples)
            indices= random.sample(range(0, len(train_samples)), sub_samples)
            sub_sampler = SubsetRandomSampler(indices)
            sampler= SequentialDistributedSampler(sub_sampler)

            dev_dataloader = DataLoader(train_samples, batch_size=batch_size_val,sampler= sampler,collate_fn = self.smart_batching_collate,shuffle=False)

            self.model.eval()
            self.model.to(self.device)

            pred_scores = []
            labels_val=[]
            loss_value=[]
            for _, (features,labels) in enumerate (dev_dataloader):
                    with torch.no_grad():
                        # loss, logits = self.model.forward(labels,features)
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss = loss_fct(logits, labels)
                    labels_val.extend(labels) 
                    pred_scores.extend(logits)
                    loss_value.append(loss) 

            pred_scores = distributed_concat(torch.stack(pred_scores),  len(indices))
            labels_val = distributed_concat(torch.stack(labels_val),  len(indices))
            loss_value = distributed_concat(torch.stack(loss_value),  len(indices))

            
            if self.master_process:
                pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
                labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])

                # pd.DataFrame(labels_val).to_csv(output_path + 'labels_train.csv', index=None, header=None)
                # pd.DataFrame(pred_scores).to_csv(output_path + 'pred_scores_train.csv', index=None, header=None)
                        
                loss = torch.mean(loss_value)
                ap_score = average_precision_score(labels_val, pred_scores)
                csv_headers = ['training_steps', "train_loss", "precision-recall auc"]
                csv_path = os.path.join(output_path + 'predict_metrics_train.csv')
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(csv_headers)
                    writer.writerow([training_steps,loss.item(),ap_score]) 

    def eval(self,args,
            dev_samples:List[InputExample],
            batch_size_val: int = 32,  
            training_steps:int=1,
            sub_samples=100,
            output_path: str = None,
            activation_fct = None,
            loss_fct=None
            ):

            if activation_fct is None:
                activation_fct = self.default_activation_function

            indices= random.sample(range(0, len(dev_samples)), sub_samples)
            sub_sampler = SubsetRandomSampler(indices)

            sampler= SequentialDistributedSampler(sub_sampler)
            
            dev_dataloader = DataLoader(dev_samples, batch_size=batch_size_val,sampler= sampler,collate_fn = self.smart_batching_collate,shuffle=False)

            self.model.eval()
            self.model.to(self.device)

            pred_scores = []
            labels_val=[]
            loss_value=[]
            for _, (features,labels) in enumerate (dev_dataloader):
                    with torch.no_grad():
                        # loss, logits = self.model.forward(labels,features)
                        model_predictions = self.model(**features, return_dict=True)
                        logits = activation_fct(model_predictions.logits)
                        if self.config.num_labels == 1:
                            logits = logits.view(-1)
                        loss = loss_fct(logits, labels)
                    labels_val.extend(labels) 
                    pred_scores.extend(logits)
                    loss_value.append(loss) 

            pred_scores = distributed_concat(torch.stack(pred_scores),  len(indices))
            labels_val = distributed_concat(torch.stack(labels_val),  len(indices))
            loss_value = distributed_concat(torch.stack(loss_value),  len(indices))
            
            if self.master_process:
                pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
                labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])
                loss = torch.mean(loss_value)

                # print(pred_scores)
                ap_score = average_precision_score(labels_val, pred_scores)

                csv_headers = ['training_steps', "val_loss", "precision-recall auc"]
                csv_path = os.path.join(output_path + 'predict_metrics_val.csv')
                output_file_exists = os.path.isfile(csv_path)
                with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not output_file_exists:
                        writer.writerow(csv_headers)
                    writer.writerow([training_steps,loss.item(),ap_score])  
    
    def predict(self, args,
               hostname:str=None,
               batch_size_val:int=32,
               training_steps=1,
               sub_samples=100,
               output_path:str=None,
               activation_fct = None,
               loss_fct=None
               ):

        if activation_fct is None:
                activation_fct = self.default_activation_function


        test_samples = load_val_objs(args.test_filepath)
    
        indices= random.sample(range(0, len(test_samples)), sub_samples)
        sub_sampler = SubsetRandomSampler(indices)
        test_sampler = SequentialDistributedSampler(sub_sampler)
        dev_dataloader = DataLoader(test_samples, batch_size=batch_size_val,sampler= test_sampler,collate_fn = self.smart_batching_collate,shuffle=False)

        self.model.eval()
        self.model.to(self.device)

        pred_scores = []
        labels_val=[]
        loss_value=[]

        for _, (features,labels) in enumerate (dev_dataloader):
                with torch.no_grad():
                    model_predictions = self.model(**features, return_dict=True)
                    logits = activation_fct(model_predictions.logits)
                    if self.config.num_labels == 1:
                            logits = logits.view(-1)
                    loss = loss_fct(logits, labels)

                loss_value.append(loss)
                pred_scores.extend(logits)
                labels_val.extend(labels) 

        pred_scores = distributed_concat(torch.stack(pred_scores),  len(test_sampler.dataset))
        labels_val = distributed_concat(torch.stack(labels_val),  len(test_sampler.dataset))
        loss_value = distributed_concat(torch.stack(loss_value),  len(test_sampler.dataset))

        if self.master_process:
            pred_scores = np.asarray([score.cpu().detach().numpy() for score in pred_scores])
            labels_val = np.asarray([label.cpu().detach().numpy() for label in labels_val])
            loss = torch.mean(loss_value)

            ap_score = average_precision_score(labels_val, pred_scores)

            csv_headers = ['training_steps',"test_loss", "precision-recall auc"]
            csv_path = os.path.join(output_path + 'predict_metrics_test.csv')
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(csv_headers)
                writer.writerow([training_steps, loss.item(), ap_score])  


def main(args,argsDict):
    #### Just some code to print debug information to stdout
    seed_offset,ddp_rank,ddp_local_rank,device= ddp_setup()
    init_process_group(backend='nccl')
    torch.cuda.set_device(device)

    if args.seed is not None:
        random.seed(args.seed)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            level=logging.INFO,
                            handlers=[LoggingHandler()])

    offline = args.offline_model_path
    model_path = offline+ args.model_name
    output_path=args.output_filepath
    model_save_path = output_path + args.task_name +'_'+ args.model_name.replace("/", "-")+'-'+ datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + args.data + '/'

    trainer = CrossEncoder(model_path, num_labels=1, max_length=args.max_length,checkpoint = args.resume_from_checkpoint)

    train_samples = load_train_objs(args.train_filepath)
    dev_samples = load_val_objs(args.dev_filepath)
    train_dataloader = DataLoader(train_samples,batch_size=args.batch_size_train,sampler =  DistributedSampler(train_samples))

    trainer.train(args,train_dataloader=train_dataloader,
            train_samples=train_samples,
            dev_samples=dev_samples,
            batch_size_train=args.batch_size_train,
            batch_size_val=args.batch_size_val,
            save_every_steps=args.save_every_steps,
            epochs = args.epochs,
            warmup_steps= args.warmup_steps,
            evaluation_steps=args.evaluation_steps,
            eval_iters=args.eval_iters,
            output_path= model_save_path,
            use_amp=True,
            gradient_accumulation_steps=args.gradient_accumulation_steps
            )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('--epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('--offline_model_path', type=str, help='offline model path')

    parser.add_argument('--save_every_steps', type=int, default=1000,help='save every && step')
    parser.add_argument("--resume_from_checkpoint",type=str,default=None,help="If the training should continue from a checkpoint folder.")

    parser.add_argument('--seed', type=int, help='seed')
    parser.add_argument('--data', type=str, help='data')
    parser.add_argument('--task_name', type=str, help='task_name')

    parser.add_argument('--batch_size_train', default=16, type=int, help='Input train batch size on each device (default: 16)')
    parser.add_argument('--batch_size_val', default=32, type=int, help='Input train batch size on each device (default: 32)')
    parser.add_argument('--train_filepath', type=str, help='train_filepath')
    parser.add_argument('--dev_filepath', type=str, help='dev_filepath')
    parser.add_argument('--test_filepath', type=str, help='test_filepath')
    parser.add_argument('--output_filepath', type=str, help='output_filepath')

    parser.add_argument('--model_name', type=str, help='model_name')
    parser.add_argument('--embedding_size', type=int, help='embedding_size')
    parser.add_argument('--warmup_steps', default=2000,type=int, help='warmup_steps')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='gradient_accumulation_steps')
    parser.add_argument('--max_length', type=int, help='max_length')
    parser.add_argument('--evaluation_steps', type=int, help='evaluation_steps')
    parser.add_argument('--eval_iters', default=100, type=int, help='eval_iters')
    parser.add_argument('--sub_samples', default=128, type=int, help='sub_samples')  

    args = parser.parse_args()
    argsDict= args.__dict__
    main(args,argsDict)
