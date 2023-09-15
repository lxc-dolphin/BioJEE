from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import subprocess
from pathlib import Path
import pickle
import sys, os
import argparse
from tqdm import tqdm,trange
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import random
import torch
import shutil
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import glob
from torch.nn import Parameter
import math
import time
import re
import copy
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from functools import partial
# from utils import timer
from models import PreMultitaskClassBase,BertMultitaskClassifier
import pdb
from optimization import *
import logging
# torch.autograd.set_detect_anomaly(True)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import json

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    # BertConfig,
    # BertForSequenceClassification,
    # BertTokenizer,
    # DistilBertConfig,
    # DistilBertForSequenceClassification,
    # DistilBertTokenizer,
    # XLMConfig,
    # XLMForSequenceClassification,
    # XLMTokenizer,
    get_linear_schedule_with_warmup,
)

import logging
logger = logging.getLogger(__name__)

from modeling_bert import BertModel
from tokenization_bert import BertTokenizer
from configuration_bert import BertConfig

from loader import prepdata
from loader.prepNN import prep4nn
from utils1 import utils1


MODEL_CLASSES = {
    "bert": (BertConfig, BertModel, BertTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _training(model,train_dataloader, nntrain_data, train_data, 
              dev_dataloader, nndev_data, dev_data,
              gold, args, tokenizer, model_encode):
        
    model.train()
    
    if args.cuda:
        print("using cuda device: %s" % torch.cuda.current_device())
        assert torch.cuda.is_available()
        model.cuda()

    
    
    
    
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    
    
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    criterion_e = nn.CrossEntropyLoss()
    loss_bce_event = nn.BCEWithLogitsLoss()
    
    
    
    t_total = len(train_dataloader) * args.epochs
    
    
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    
    loss_train_tt = []
    metrics_dev = []
    
    dataset_size = len(train_dataloader)
    
    
     # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataloader))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Train batch size = %d",
        args.train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    
    global_step = 0
    epochs_trained = 0
    
    
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    
    # flat and nested event structure maps
    EV_structure = OrderedDict()
    EV_structure['Negative_regulation'] = OrderedDict()
    EV_structure['Negative_regulation']['Theme0Any'] = [(1,0)]
    EV_structure['Negative_regulation']['Theme0Any+Cause0Any'] = [(1,0),(2,0)]
    EV_structure['Negative_regulation']['Theme0Any+Site0Entity'] = [(1,0),(3,11)]
    EV_structure['Negative_regulation']['Theme0Any+Cause0Any+Site0Entity'] = [(1,0),(2,0),(3,11)]
    
    EV_structure['Gene_expression'] = OrderedDict()
    EV_structure['Gene_expression']['Theme0Entity'] = [(1,11)]
    
    EV_structure['Regulation'] = OrderedDict()
    EV_structure['Regulation']['Theme0Any'] = [(1,0)]
    EV_structure['Regulation']['Theme0Any+Cause0Any'] = [(1,0),(2,0)]
    EV_structure['Regulation']['Theme0Any+Site0Entity'] = [(1,0),(3,11)]
    EV_structure['Regulation']['Theme0Any+Cause0Any+Site0Entity'] = [(1,0),(2,0),(3,11)]
    
    EV_structure['Transcription'] = OrderedDict()
    EV_structure['Transcription']['Theme0Entity'] = [(1,11)]
    
    EV_structure['Positive_regulation'] = OrderedDict()
    EV_structure['Positive_regulation']['Theme0Any'] = [(1,0)]
    EV_structure['Positive_regulation']['Theme0Any+Cause0Any'] = [(1,0),(2,0)]
    EV_structure['Positive_regulation']['Theme0Any+Site0Entity'] = [(1,0),(3,11)]
    EV_structure['Positive_regulation']['Theme0Any+Cause0Any+Site0Entity'] = [(1,0),(2,0),(3,11)]
    
    EV_structure['Binding'] = OrderedDict()
    EV_structure['Binding']['Theme0Entity'] = [(1,11)]
    EV_structure['Binding']['Theme0Entity+Site0Entity'] = [(1,11),(2,11)]
    
    EV_structure['Localization'] = OrderedDict()
    EV_structure['Localization']['Theme0Entity'] = [(1,11)]
    EV_structure['Localization']['Theme0Entity+AtLoc0Entity'] = [(1,11),(5,11)]
    EV_structure['Localization']['Theme0Entity+ToLoc0Entity'] = [(1,11),(4,11)]
    EV_structure['Localization']['Theme0Entity+AtLoc0Entity+ToLoc0Entity'] = [(1,11),(5,11),(4,11)]
    
    EV_structure['Phosphorylation'] = OrderedDict()
    EV_structure['Phosphorylation']['Theme0Entity'] = [(1,11)]
    EV_structure['Phosphorylation']['Theme0Entity+Site0Entity'] = [(1,11),(3,11)]
    
    EV_structure['Protein_catabolism'] = OrderedDict()
    EV_structure['Protein_catabolism']['Theme0Entity'] = [(1,11)]
    
    args.ev_weight = 1.0
    
    best_F1 = -1.0
    
    for epoch_num in train_iterator:

        predicted_interactions = []
        predicted_interaction_labels = []
        predicted_entities = []
        all_gold_interactions = [] # during training the gold and prediction are the same
        all_gold_interaction_labels = []
        all_gold_entities = []
        all_input_ids = []
        all_sample_ids = []

        y_trues_e, y_preds_e = [], []
        y_trues_r, y_preds_r = [], []
        
        
        tr_loss = 0.0
        
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        for step, batch in enumerate(epoch_iterator):
            
            
            model.train()
            model_encode.eval()
            
            # break
            
            batch = tuple(t.to(args.device) for t in batch)

            # batch[0] = torch.tensor([257])
            
            train_input_ids = nntrain_data['nn_data']['ids'][batch[0]]
            train_input_ids = torch.tensor([train_input_ids],dtype = torch.long).to(args.device)
            
            train_input_masks = nntrain_data['nn_data']['attention_mask'][batch[0]]
            train_input_masks = torch.tensor([train_input_masks], dtype=torch.long).to(args.device)
            
            train_entity_labels = nntrain_data['nn_data']['entity_label'][batch[0]]
            train_entity_labels = torch.tensor([train_entity_labels], dtype = torch.long).to(args.device)
            
            
            train_entity_infos = nntrain_data['nn_data']['candi_info'][batch[0]]
            train_entity_infos_mask = nntrain_data['nn_data']['candi_mask'][batch[0]]
            
            
            train_entityT_posi  = nntrain_data['entityTs'][batch[0]]
        
            train_entityT_posi_list = [train_entityT_posi[_key] for _key in train_entityT_posi]
            
            
            
            # get entity infos
            out_enti_info = torch.zeros(size = (1, 512, 768), dtype = torch.float32).to(args.device)
            
            
            if args.add_candi:
                for entity_idx, entity_info_each in enumerate(train_entity_infos):
                    entity_info_each_mask = train_entity_infos_mask[entity_idx]
                    
                    entity_posi_start = train_entityT_posi_list[entity_idx][0]
                    entity_posi_end = train_entityT_posi_list[entity_idx][1]
                    
                    entity_info_each = torch.tensor([entity_info_each], dtype = torch.long).to(args.device)
                    entity_info_each_mask = torch.tensor([entity_info_each_mask], dtype = torch.long).to(args.device)
                    with torch.no_grad():
                        out_temp = model_encode.bert(entity_info_each,entity_info_each_mask,
                                                    token_type_ids=None,
                                                    position_ids=None,
                                                    head_mask=None)
                        out_enti_each = torch.cat((entity_posi_end-entity_posi_start+1)*[out_temp[1].unsqueeze(1)], dim = 1)
                    
                    
                    
                    out_enti_info[0, entity_posi_start+1:entity_posi_end+2,:] = out_enti_info[0, entity_posi_start+1:entity_posi_end+2,:] + out_enti_each
                # 1 for cls token, 1+1 for cls and slicing function
            
            out_enti_info = out_enti_info.to(args.device) 
            ###############entity output#########################
            out_e, prob_e, sentence_embeds = model.forward(train_input_ids, train_entity_labels, 
                                          train_input_masks,out_enti_info,
                                          task = 'entity',args = args)

            # mask out the prob of the padding with input mask        
            mask = torch.cat( [train_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
            # mask[:,:,0] = 1
            prob_e *= mask
            
            # construct relation
            
            train_arg_rela = nntrain_data['nn_data']['tri_arg_rela'][batch[0]] # (trigger, entity) + (trigger, trigger)
            train_arg_rela_label = nntrain_data['nn_data']['tri_arg_rela_label'][batch[0]] # corresponding paired labels
            
            
            train_interaction = nntrain_data['nn_data']['tri_arg_interaction'][batch[0]] # token-wised trigger and arguement
            train_interaction_label = nntrain_data['nn_data']['tri_arg_interaction_label'][batch[0]] # token_wised trigger and argument labels
            # if train_arg_rela != []:
            #     if len(train_arg_rela[0]) == 2:
            #         print(train_arg_rela)
            #         print(train_arg_rela_label)
                
            
            
            label_r, rel_idxs, lidx, ridx = model.construct_relations(prob_e, train_entity_labels, train_input_masks, 
                                                                      train_interaction, train_interaction_label, 
                                                                      args, gold=gold, test=False)
            
            assert len(lidx) == len(ridx) # assert event numbers are equal (in trigger and arg )
            # retrieve the prediected pairs
            pair_lengths = [len(i) for i in lidx]  # num of pairs in each event 
            for i in range(len(lidx)): # batch size, event number
                if len(lidx[i]) == 0:
                    predicted_interactions.append([])
                else:
                    assert len(lidx[i]) == len(ridx[i])
                    predicted_interactions.append([i for i in zip(lidx[i], ridx[i])]) # by event: (trig_posi,arg_posi)
            
            if lidx == []:
                predicted_interactions.append([])

            ############### predict relations####################
            if rel_idxs != []: # predicted relation could be empty --> skip

                    
                out_r, prob_r = model.forward(train_input_ids, train_entity_labels, 
                                              train_input_masks,out_enti_info,
                                              arg_rela = train_interaction, arg_rela_label = train_interaction_label, 
                                              rel_idxs=rel_idxs, lidx=lidx, ridx=ridx, task='relation', args=args)
                
                # (batch, )
                     
                pred_r = prob_r.data.argmax(dim=1).long().view(-1)
                
                # if step == 82:
                #     print(pred_r)
                #     print(label_r)
                #     print(rel_idxs)
                #     print(lidx)
                #     print(ridx)
                #     print(train_interaction)
                #     print(train_interaction_label)
                
                assert pred_r.size(0) == label_r.size(0)

                if args.cuda:
                    prob_r = prob_r.cpu()
                    
                    # label_r = label_r.cpu()
                
                pred_r_list = pred_r.tolist()
                # extend to all predicted relations
                y_preds_r.extend(pred_r_list)

                # # retrive the ints labels for the predicted pairs
                # cur = 0
                # for i, l in enumerate(pair_lengths): # to each event, i:id, l: trig and arg pair number
                #     if pair_lengths[i] == 0:
                #         predicted_interaction_labels.append([])
                #     else:
                #         for j in range(l): # each trig and arg
                #             for jj in range(len(predicted_interactions[i][j])):
                                
                #                 predicted_interaction_labels.append([args._id_to_label_i[x] for x in pred_r_list[cur:cur+l]])
                #                 cur += l

                
                predicted_interaction_labels.append([args._id_to_label_ii[x] for x in pred_r_list])
                # if step== 152:
                #     print(label_r)
                #     print(label_r.size())
                    
                all_gold_interaction_labels.append([args._id_to_label_ii[x] for x in label_r.tolist()])
                
            else: # no relation predicted

                y_preds_r.extend([])
                predicted_interaction_labels.extend([])
                # predicted_interaction_labels.extend([[] for _ in range(len(train_input_masks))])

            # assert len(predicted_interaction_labels[-1]) ==len(predicted_interactions[-1])
            
            y_trues_r.extend(label_r.tolist())            
            
            # retrieve and flatten entity prediction for loss calculation
            ent_pred, ent_label, ent_out, ent_key, ent_pos, ent_input = [], [], [], [], [], []

            # get entities prediction filtered by mask
            for i, mask in enumerate(train_input_masks):
                
                mask = mask.bool()
                # take only mask==1 portion
                ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
                
                # flatten entity label
                ent_label.append(torch.masked_select(train_entity_labels[i], mask))
                
                ent_input.append(torch.masked_select(train_input_ids[i], mask))

                all_gold_entities.append(ent_label[-1].tolist())
                predicted_entities.append(ent_pred[-1].tolist())
                all_input_ids.append(ent_input[-1].tolist())
                
            ## collect trigger prediction results
            ent_pred = torch.cat(ent_pred, 0)
            ent_label = torch.cat(ent_label, 0)
            
            
            
            assert ent_pred.size() == ent_label.size() 


            y_trues_e.extend(ent_label.tolist())
            y_preds_e.extend(ent_pred.tolist())
            
            # calculate loss
            mask_e_out = torch.cat( [train_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
            
            # if args.cuda:
            #     out_e = out_e.cpu()
            
            out_e_toloss = torch.masked_select(out_e,mask_e_out.bool())
            
            
            loss_e = criterion_e(out_e_toloss.reshape((ent_label.shape[0],-1)), ent_label)
            
            if rel_idxs != []:
                
                loss_r = criterion_e(out_r, label_r.to(args.device))            
            else:
                loss_r = 0

            loss_ev = 0.0 
            loss_tt = args.entity_weight * loss_e + args.relation_weight * loss_r + args.ev_weight * loss_ev * 0.0
            
            
            
            loss_tt.backward()    
            
            tr_loss += loss_tt.item()
            
            
            if ((global_step+1) % args.gradient_accumulation_steps == 0) or (step+1 == len(train_dataloader)):
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
            global_step += 1
        
        
        result, dev_metric = evaluate(model, dev_dataloader,nndev_data, dev_data,
                                        gold, args,tokenizer, model_encode, prefix=prefix)
            
        print(result)
        print(dev_metric)
        
        metrics_dev.append(dev_metric)
        
        curt_F1 = dev_metric[5]
        
        if True: #curt_F1 > best_F1
        
            # Save model checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = (
                model.module if hasattr(model, "module") else model
            )  # Take care of distributed/parallel training
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)

            torch.save(args, os.path.join(output_dir, "training_args.bin"))
            logger.info("Saving model checkpoint to %s", output_dir)

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if curt_F1 > best_F1:
                best_F1 = curt_F1
        
        loss_train_tt.append(tr_loss/dataset_size)
            
        # if args.max_steps > 0 and global_step > args.max_steps:
        #     train_iterator.close()
        #     break
    
    
    
    return loss_train_tt, metrics_dev



def _evaluation_intrain(model, dev_dataloader, 
                        nndev_data, dev_data,
                        gold, args, tokenizer):
    
    dev_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    
    
    
    arg_tp, arg_tn, arg_fp, arg_fn = 0, 0, 0, 0
    tri_tp, tri_tn, tri_fp, tri_fn = 0, 0, 0, 0
    
    
    for step, batch in enumerate(dev_iterator):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
            
        dev_input_ids = nndev_data['nn_data']['ids'][batch[0]]
        dev_input_ids = torch.tensor([dev_input_ids],dtype = torch.long).to(args.device)
        
        dev_input_masks = nndev_data['nn_data']['attention_mask'][batch[0]]
        dev_input_masks = torch.tensor([dev_input_masks], dtype=torch.long).to(args.device)
        
        dev_entity_labels = nndev_data['nn_data']['entity_label'][batch[0]]
        dev_entity_labels = torch.tensor([dev_entity_labels], dtype = torch.long).to(args.device)
        
        ## entity estimation ##
        out_e, prob_e = model.forward(dev_input_ids, dev_entity_labels, 
                                    dev_input_masks,
                                    task = 'entity',args = args)
        # mask out the prob of the padding with input mask        
        mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
        prob_e *= mask
        
        # construct relation
        dev_arg_rela = nndev_data['nn_data']['tri_arg_rela'][batch[0]]
        dev_arg_rela_label = nndev_data['nn_data']['tri_arg_rela_label'][batch[0]]
        
        label_r, rel_idxs, lidx, ridx = model.construct_relations(prob_e, dev_entity_labels, dev_input_masks, 
                                                                dev_arg_rela, dev_arg_rela_label, 
                                                                args, gold=gold, test=False)
        # lidx: trigger posi, ridx: arg posi    
        assert len(lidx) == len(ridx) # assert event numbers are equal (in trigger and arg )
        
        trig_arg_offset = [] # get the relative offset number/length from each trig arg pair 
        t_offset = 0
        for evid, _ in enumerate(lidx): # by event
            tri_posi_temp = lidx[evid]
            arg_posi_temp = ridx[evid]
            for argid, _ in enumerate(tri_posi_temp): # by each trig arg pair
                trig_arg_offset.append(len(tri_posi_temp[argid]))
                t_offset += len(tri_posi_temp[argid])
                trig_arg_offset.append(len(arg_posi_temp[argid]))
                t_offset += len(arg_posi_temp[argid])
        assert t_offset == len(label_r.tolist())
        
        ## predict relation ## and calculate metrics for relation/argument
        if rel_idxs != []:
            out_r, prob_r = model.forward(dev_input_ids, dev_entity_labels, 
                                        dev_input_masks,
                                        arg_rela = dev_arg_rela, arg_rela_label = dev_arg_rela_label, 
                                        rel_idxs=rel_idxs, lidx=lidx, ridx=ridx, task='relation', args=args)
                
            pred_r = prob_r.data.argmax(dim=2).long().view(-1)
                
            assert pred_r.size(0) == label_r.size(0)
        
            # if args.cuda:
                    # prob_r = prob_r.cpu()
                    
                    # label_r = label_r.cpu()
        
            pred_r_list = pred_r.tolist()
            label_r_list = label_r.tolist()
            cur = 0
            for idx, _off_each in enumerate(trig_arg_offset):
                
                if idx%2 == 1: # arg_offset
                    lab_p = 0
                    lab_t = 0
                    
                    # calculate label by word not token
                    for _lab in pred_r_list[cur:cur+_off_each]:
                        lab_p += _lab
                    for _lab in label_r_list[cur:cur+_off_each]:
                        lab_t += _lab
                    
                    lab_p = lab_p/_off_each # mean label value for prediction
                    lab_t = lab_t/_off_each
                        
                    if lab_p < 0.5 and lab_t > 0.5:
                        arg_fn += 1
                        break
                    elif lab_p < 0.5 and lab_t < 0.5:
                        arg_tn += 1
                        break
                    elif abs(lab_p-lab_t) < 0.5:
                        arg_tp += 1
                    elif abs(lab_p-lab_t) >= 0.5:
                        arg_fp += 1        
                cur += _off_each
            
            
        # calculate entity estimation metircs     
        # get entities prediction filtered by mask
        ent_pred = []
        ent_label = []
        
        terms_evs = nndev_data['terms_evs'][batch[0]]
        readable_ents = dev_data[batch[0]]['readable_ents']
        if terms_evs != {}:
            for i, mask in enumerate(dev_input_masks):
                
                mask = mask.bool()
                # take only mask==1 portion
                ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
                
                # flatten entity label
                ent_label.append(torch.masked_select(dev_entity_labels[i], mask))
            
            for evid in terms_evs:
                trig_pre = 0
                trig_gold = 0
                # get trigger word tag
                trig_tag = terms_evs[evid][2]
                toks_trig = readable_ents[trig_tag]['toks']
                toks_trig_posi  = [posi_+1 for posi_ in toks_trig]
                
                #get label scores for each trigger words
                for posi_ in toks_trig_posi:
                    trig_pre += ent_pred[0][posi_].tolist()
                    trig_gold += ent_label[0][posi_].tolist()

                trig_pre = trig_pre/len(toks_trig_posi)
                trig_gold = trig_gold/len(toks_trig_posi)
                
                if trig_pre < 0.5 and trig_gold > 0.5:
                    tri_fn += 1
                    break
                elif trig_pre < 0.5 and trig_gold < 0.5:
                    tri_tn += 1
                    break
                elif abs(trig_pre-trig_gold) < 0.5:
                    tri_tp += 1
                elif abs(trig_pre-trig_gold) >= 0.5:
                    tri_fp += 1
            
            
            
    recall_tri, precision_tri, f_score_tri = tri_tp/(tri_tp+tri_fn), tri_tp/(tri_tp+tri_fp), 2*tri_tp/(tri_tp*2+tri_fp+tri_fn) 
    recall_arg, precision_arg, f_score_arg = arg_tp/(arg_tp+arg_fn), arg_tp/(arg_tp+arg_fp), 2*arg_tp/(arg_tp*2+arg_fp+arg_fn)
            
    return [recall_tri, precision_tri, f_score_tri, recall_arg, precision_arg, f_score_arg]


def evaluate(model, dev_dataloader,nndev_data, dev_data,
            gold, args,tokenizer, model_encode, prefix=None):
    result = {}
    # dev_metric = []
    
    dev_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    
    
    
    arg_tp, arg_tn, arg_fp, arg_fn = 0, 0, 0, 0
    
    tri_tp, tri_tn, tri_fp, tri_fn = 0, 0, 0, 0
    
    
    type_match_gold, type_false_positive, type_false_nagetive = {}, {}, {}
    
    event_types = set([
    "Binding",
    "Gene_expression",
    "Localization",
    "Transcription",
    "Phosphorylation",
    "Protein_catabolism",
    "Regulation",
    "Negative_regulation",
    "Positive_regulation",
    ])
    
    #initialize the type errors    
    for etype in event_types:
        type_match_gold[etype] = 0
        type_false_positive[etype] = 0
        type_false_nagetive[etype] = 0
    
    
    
    just_trig = 0
    
    for step, batch in enumerate(dev_iterator):
        
        model.eval()
        model_encode.eval()
        
        batch = tuple(t.to(args.device) for t in batch)
            
        dev_input_ids = nndev_data['nn_data']['ids'][batch[0]]
        dev_input_ids = torch.tensor([dev_input_ids],dtype = torch.long).to(args.device)
        
        dev_input_masks = nndev_data['nn_data']['attention_mask'][batch[0]]
        dev_input_masks = torch.tensor([dev_input_masks], dtype=torch.long).to(args.device)
        
        dev_entity_labels = nndev_data['nn_data']['entity_label'][batch[0]]
        dev_entity_labels = torch.tensor([dev_entity_labels], dtype = torch.long).to(args.device)
        
        
        dev_entity_infos = nndev_data['nn_data']['candi_info'][batch[0]]
        dev_entity_infos_mask = nndev_data['nn_data']['candi_mask'][batch[0]]
        
        dev_entityT_posi  = nndev_data['entityTs'][batch[0]]
        
        dev_entityT_posi_list = [dev_entityT_posi[_key] for _key in dev_entityT_posi]
             
                
        
        # get entity infos
        out_enti_info = torch.zeros(size = (1, 512, 768), dtype = torch.float32).to(args.device)
        
        if args.add_candi:
            for entity_idx, entity_info_each in enumerate(dev_entity_infos):
                entity_info_each_mask = dev_entity_infos_mask[entity_idx]
                
                entity_posi_start = dev_entityT_posi_list[entity_idx][0]
                entity_posi_end = dev_entityT_posi_list[entity_idx][1]
                
                entity_info_each = torch.tensor([entity_info_each], dtype = torch.long).to(args.device)
                entity_info_each_mask = torch.tensor([entity_info_each_mask], dtype = torch.long).to(args.device)
                with torch.no_grad():
                    out_temp = model_encode.bert(entity_info_each,entity_info_each_mask,
                                                token_type_ids=None,
                                                position_ids=None,
                                                head_mask=None)
                    out_enti_each = torch.cat((entity_posi_end-entity_posi_start+1)*[out_temp[1].unsqueeze(1)], dim = 1)
                
                
                
                out_enti_info[0, entity_posi_start+1:entity_posi_end+2,:] = out_enti_info[0, entity_posi_start+1:entity_posi_end+2,:] + out_enti_each
             # 1 for cls token, 1+1 for cls and slicing function
            # temp_prefix = torch.zeros(size = (1,entity_posi_start+1,768), dtype=torch.float32)   
            # temp_suffix = torch.zeros(size = (1,512-(entity_posi_start+1)-(entity_posi_end-entity_posi_start+1),768), dtype=torch.float32)
            
            # temp_enti_each = torch.cat((temp_prefix, out_enti_each, temp_suffix), dim = 1)
            
            # out_enti_info = out_enti_info + temp_enti_each    
        
        
        out_enti_info = out_enti_info.to(args.device)   
        ## entity estimation ##
        out_e, prob_e, out = model.forward(dev_input_ids, dev_entity_labels, 
                                    dev_input_masks, out_enti_info,
                                    task = 'entity',args = args)
        # mask out the prob of the padding with input mask        
        mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
        prob_e *= mask
        
        
        
        # construct relation
        dev_arg_rela = nndev_data['nn_data']['tri_arg_rela'][batch[0]]
        dev_arg_rela_label = nndev_data['nn_data']['tri_arg_rela_label'][batch[0]]
        
        
        dev_interaction = nndev_data['nn_data']['tri_arg_interaction'][batch[0]] # token-wised trigger and arguement
        dev_interaction_label = nndev_data['nn_data']['tri_arg_interaction_label'][batch[0]] # token_wised trigger and argument labels
            
        
        
        label_r, rel_idxs, lidx, ridx = model.construct_relations(prob_e, dev_entity_labels, dev_input_masks, 
                                                                dev_interaction, dev_interaction_label, 
                                                                args, gold=gold, test=False, do_eval=True)
        # lidx: trigger posi, ridx: arg posi    
        assert len(lidx) == len(ridx) # assert event numbers are equal (in trigger and arg )
        
        trig_arg_offset = [] # get the trig and arg offset from each trig arg pair that should check/evaluate 
        # t_offset = 0
        for evid, _ in enumerate(dev_arg_rela): # by event
            tri_arg_posi_temp = dev_arg_rela[evid] # paired posi [trig,arg1], [trig,arg2], ... 
            # tri_arg_posi_label_temp= dev_arg_rela_label[evid] # lables [trig,arg1], [trig, arg2],...
            
            for pair_id, _ in enumerate(tri_arg_posi_temp): # by each trig arg pair
                trig_arg_offset.append((tri_arg_posi_temp[pair_id][0][0],tri_arg_posi_temp[pair_id][1][0]))
    
        # assert t_offset == len(label_r.tolist())
        
        
        
        
        ## predict relation ## and calculate metrics for relation/argument
        if rel_idxs != []:
            out_r, prob_r = model.forward(dev_input_ids, dev_entity_labels, 
                                        dev_input_masks, out_enti_info,
                                        arg_rela = dev_interaction, arg_rela_label = dev_interaction_label, 
                                        rel_idxs=rel_idxs, lidx=lidx, ridx=ridx, task='relation', args=args)
                
            pred_r = prob_r.data.argmax(dim=1).long().view(-1)
                
            assert pred_r.size(0) == label_r.size(0)
        
            # if args.cuda:
                    # prob_r = prob_r.cpu()
                    
                    # label_r = label_r.cpu()
        
            pred_r_list = pred_r.tolist()
            label_r_list = label_r.tolist()
            
            # compare each event: triger and arguments
            ent_pred = []
            ent_label = []
            
            terms_evs = nndev_data['terms_evs'][batch[0]]
            readable_ents = dev_data[batch[0]]['readable_ents']
            terms_enti = nndev_data['termss'][batch[0]] # including entity and trigger
            
            sub_to_words = nndev_data['sub_to_words'][batch[0]] # sub word sheet to word
             
            if terms_evs != {}: # compare trigger in the sentence
                for i, mask in enumerate(dev_input_masks):
                    
                    mask = mask.bool()
                    # take only mask==1 portion
                    ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
                    
                    # flatten entity label
                    ent_label.append(torch.masked_select(dev_entity_labels[i], mask))

                
                ent_pred = ent_pred[0].tolist()
                ent_label = ent_label[0].tolist()
                
                
                sub_lab_id = 0
                for lab_num_, lab_id_ in enumerate(ent_pred):
                    sub_lab_id += abs(ent_pred[lab_num_]-ent_label[lab_num_])

                if (sub_lab_id > 1):
                    tri_fp += 1

                # find the redundant event in predictions (false positive)
                pre_event_temp = {} # {trig_start_posi: label,...}
                
                ent_pred_np = np.array(ent_pred)
                protein_id = 10
                entity_id = 11
                pre_tri_idx = np.where((ent_pred_np> 0) *( ent_pred_np != protein_id)*(ent_pred_np != entity_id))[0].tolist()
                
                cur_ = -1
                for tri_posi in pre_tri_idx:
                    
                    if tri_posi > cur_:
                        if tri_posi not in pre_event_temp:                        
                            _tri_start_ = tri_posi
                            pre_event_temp[_tri_start_] = ent_pred[_tri_start_]
                        
                        cur_ = tri_posi
                        
                        if cur_ < len(sub_to_words):    
                            while sub_to_words[cur_-1] == sub_to_words[cur_]:
                                cur_ += 1
                                if cur_ >= len(sub_to_words): 
                                    break

                 
                gold_event_temp = {} # get gold event start trigger posi {trig_start_posi : label,...}
                for evid_ in terms_evs: # each event
                    event_info_ = terms_evs[evid_]
                    trig_tag_ = event_info_[2] # 'TR30' or others like 'T17' 
                    posi_tri_start_ = 0
                    posi_tri_end_ = 0
                    for star_end_ in terms_enti: # get the trigger position
                        if terms_enti[star_end_][0] == trig_tag_:
                            posi_tri_start_ = star_end_[0]+1
                            posi_tri_end_ = star_end_[1]+1
                    if posi_tri_end_ == 0:
                        
                        break # didn't find any triggers
                    
                    if posi_tri_start_ not in gold_event_temp:
                        gold_event_temp[posi_tri_start_] = args._id_to_label_t[ent_label[posi_tri_start_]]
                
                # identify the triggers are not in the gold event
                for key_ in pre_event_temp:
                    if key_ not in gold_event_temp:
                        label_temp = pre_event_temp[key_]
                        type_false_positive[args._id_to_label_t[label_temp]] += 1
                        
                        for key_gold_ in gold_event_temp:
                            type_false_nagetive[gold_event_temp[key_gold_]] += 0.5
                
                
                
                    
                    
                # compare with gold
                #  for each event compare the trigger/argu position and lables 
                ev_num = 0
                for evid_ in terms_evs: # each event
                    event_info_ = terms_evs[evid_]
                    trig_tag_ = event_info_[2] # 'TR30' or others like 'T17' 
                    posi_tri_start_ = 0
                    posi_tri_end_ = 0
                    for star_end_ in terms_enti: # get the trigger position
                        if terms_enti[star_end_][0] == trig_tag_:
                            posi_tri_start_ = star_end_[0]+1
                            posi_tri_end_ = star_end_[1]+1
                    if posi_tri_end_ == 0:
                        
                        break # didn't find any triggers  
                    sub_lab_id = 0
                    for lab_num_, lab_id_ in enumerate(ent_pred):
                        sub_lab_id += abs(ent_pred[lab_num_]-ent_label[lab_num_])
                    
                    
                    gold_ev_type = args._id_to_label_t[ent_label[posi_tri_start_]]
                    
                    pre_ev_id = args._id_to_label_t[ent_pred[posi_tri_start_]]
                    
                    if (True) and (gold_ev_type == pre_ev_id):#(ent_pred[posi_tri_start_:posi_tri_end_+1] == ent_label[posi_tri_start_:posi_tri_end_+1]):
                    # if above statement pass, then trigger matched
                    # compare the arguments
                        just_trig +=1
                        
                        if ev_num+1 <= len(dev_arg_rela):
                        
                            trig_arg_posi_pair = dev_arg_rela[ev_num]
                            arg_wrong = False
                            
                            for pair_id_,_ in enumerate(trig_arg_posi_pair):  # each event, paired posi [[trig],[arg1]], [[trig],[arg2]], ... 
                                
                                # print(pair_id_)
                                
                                trig_arg_pair_each = trig_arg_posi_pair[pair_id_]
                                trig_posi_ = trig_arg_pair_each[0]
                                arg_posi_ = trig_arg_pair_each[1]
                                
                                pair_idx = dev_interaction.index((trig_posi_[0],arg_posi_[0]))
                                batch_number = 0
                                pair_label_gold = dev_interaction_label[(trig_posi_[0],arg_posi_[0])][0]
                                pair_pre_label = pred_r_list[pair_idx]
                                
                                # 
                                if pair_pre_label != pair_label_gold:
                                    if pair_pre_label == 0:
                                        arg_fn += 1 # predict other is the argument
                                        arg_wrong = True
                                    else:
                                        arg_fp += 1 # predict wrong label
                                        arg_wrong = True
                                else:
                                    arg_tp += 1
                                    
                                    
                            if arg_wrong == False:
                                tri_tp += 1
                                type_match_gold[gold_ev_type] += 1
                            else:
                                tri_fn += 1
                                type_false_nagetive[gold_ev_type] += 1 
                                type_false_positive[pre_ev_id] += 1       
                    else:
                        id_need_test = ent_pred[posi_tri_start_:posi_tri_end_+1][0]
                        if id_need_test == 0 or id_need_test == 10 or id_need_test == 11:
                            tri_fn += 1
                            type_false_nagetive[gold_ev_type] += 1
                            
                            type_false_positive[gold_ev_type] += 1
                        else:
                            
                            tri_fp += 1
                            
                            type_false_nagetive[gold_ev_type] += 1
                            
                            type_false_positive[pre_ev_id] += 1
                            
                        
                    ev_num += 1
        
            
            
        
            
            
            
            
    recall_tri, precision_tri, f_score_tri = tri_tp/(tri_tp+tri_fn+1), tri_tp/(tri_tp+tri_fp+1), 2*tri_tp/(tri_tp*2+tri_fp+tri_fn+1) 
    recall_arg, precision_arg, f_score_arg = arg_tp/(arg_tp+arg_fn+1), arg_tp/(arg_tp+arg_fp+1), 2*arg_tp/(arg_tp*2+arg_fp+arg_fn+1)

    
    type_recall = {}
    type_precision ={}
    type_F1 = {}
    
    tt_recall, tt_precision, tt_F1 = 0.0, 0.0, 0.0
    
    for ev_t in event_types:
        type_recall[ev_t],type_precision[ev_t], type_F1[ev_t] = recall_pre_F1(type_match_gold[ev_t],
                                                                              type_false_positive[ev_t],
                                                                              type_false_nagetive[ev_t])

    
    for ev_t in event_types:
        tt_recall += type_recall[ev_t]
        tt_precision += type_precision[ev_t]
        tt_F1 += type_F1[ev_t]      
    
    print(len(event_types))
    
    tt_recall = tt_recall/len(event_types)
    tt_precision = tt_precision/(len(event_types))
    tt_F1 = tt_F1/(len(event_types))
    
    
    
    return [[just_trig, tri_tp, tri_fp, tri_fn], 
            [recall_tri, precision_tri, 
             f_score_tri, recall_arg, 
             precision_arg, f_score_arg]], [type_recall,type_precision,type_F1,tt_recall,tt_precision,tt_F1-0.01]
    
 

def recall_pre_F1(tp,fp,fn):
    
    recall = tp/(1*tp+fn)
    precision = tp/(1*tp+fp)
    F1 = 2*tp/(tp*2+fp+fn)
    
    return recall, precision, F1
# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)

# logger = logging.getLogger(__name__)

def predict(model, dev_dataloader,nndev_data, dev_data,
            gold, args,tokenizer, model_encode, prefix=None):
    result = {}
    # dev_metric = []
    
    dev_iterator = tqdm(dev_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
    
    y_preds_e = []
    

    
    for step, batch in enumerate(dev_iterator):
        
        model.eval()
        model_encode.eval()
        
        batch = tuple(t.to(args.device) for t in batch)
            
        dev_input_ids = nndev_data['nn_data']['ids'][batch[0]]
        dev_input_ids = torch.tensor([dev_input_ids],dtype = torch.long).to(args.device)
        
        dev_input_masks = nndev_data['nn_data']['attention_mask'][batch[0]]
        dev_input_masks = torch.tensor([dev_input_masks], dtype=torch.long).to(args.device)
        
        dev_entity_labels = nndev_data['nn_data']['entity_label'][batch[0]]
        dev_entity_labels = torch.tensor([dev_entity_labels], dtype = torch.long).to(args.device)
        
        
        dev_entity_infos = nndev_data['nn_data']['candi_info'][batch[0]]
        dev_entity_infos_mask = nndev_data['nn_data']['candi_mask'][batch[0]]
        
        
        # get entity infos
        out_enti_info = torch.zeros(size = (1, 1, 768), dtype = torch.long).to(args.device)
        
        if len(dev_entity_infos) == 512:
            entity_info_each = torch.tensor([dev_entity_infos], dtype = torch.long).to(args.device)
            entity_info_each_mask = torch.tensor([dev_entity_infos_mask], dtype = torch.long).to(args.device)
            with torch.no_grad():
                out_temp = model_encode.bert(entity_info_each,entity_info_each_mask,
                                            token_type_ids=None,
                                            position_ids=None,
                                            head_mask=None )
            out_enti_info  = out_enti_info + out_temp[1]
        else:
            for entity_idx, entity_info_each in enumerate(dev_entity_infos):
                entity_info_each_mask = dev_entity_infos_mask[entity_idx]
                
                entity_info_each = torch.tensor([entity_info_each], dtype = torch.long).to(args.device)
                entity_info_each_mask = torch.tensor([entity_info_each_mask], dtype = torch.long).to(args.device)
                with torch.no_grad():
                    out_temp = model_encode.bert(entity_info_each,entity_info_each_mask,
                                                token_type_ids=None,
                                                position_ids=None,
                                                head_mask=None )
                out_enti_info = out_enti_info + out_temp[1]
        
        
            
        ## entity estimation ##
        out_e, prob_e = model.forward(dev_input_ids, dev_entity_labels, 
                                    dev_input_masks, out_enti_info,
                                    task = 'entity',args = args)
        # mask out the prob of the padding with input mask        
        mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
        prob_e *= mask
        
        # get entities prediction filtered by mask
        ent_pred = []
        ent_label = []
        for i, mask in enumerate(dev_input_masks):
            
            mask = mask.bool()
            # take only mask==1 portion
            ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
            # flatten entity label
            ent_label.append(torch.masked_select(dev_entity_labels[i], mask))
        
        # print(1)
        assert ent_pred[0].size() == ent_label[0].size() 


        # y_trues_e.append(ent_label.tolist())
        y_preds_e.append(ent_pred[0].tolist())
        
    result['predict_EL'] = y_preds_e
    
    
    
        
    return result
    

def predict_single(model, dev_dataloader,nndev_data, dev_data,
            gold, args,tokenizer, model_encode, prefix=None):
    result = {}
    # dev_metric = []
    
    model.eval()
    
    # batch = tuple(t.to(args.device) for t in batch)
            
    dev_input_ids = nndev_data['ids']
    dev_input_ids = torch.tensor([dev_input_ids],dtype = torch.long).to(args.device)
    
    dev_input_masks = nndev_data['attention_mask']
    dev_input_masks = torch.tensor([dev_input_masks], dtype=torch.long).to(args.device)
    
    # dev_entity_labels = nndev_data['entity_label']
    dev_entity_labels = torch.zeros(size = (1, 512), dtype = torch.long).to(args.device) # torch.tensor([0]*512, dtype = torch.long).to(args.device)
    
    
    
    out_enti_info = torch.zeros(size = (1, 512, 768), dtype = torch.float32).to(args.device)
    
    out_e, prob_e = model.forward(dev_input_ids, dev_entity_labels, 
                                    dev_input_masks, out_enti_info,
                                    task = 'entity',args = args)
    # mask out the prob of the padding with input mask        
    mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (out_e.size(2) ),dim=2)
    prob_e *= mask
    
    # predicted_entities = prob_e.argmax(dim=2)
    
    protein_id = args._label_to_id_t['Protein'] 
    entity_id = args._label_to_id_t['Entity']
    
    ent_pred = []
    ent_label = []
    
    
    for i, mask in enumerate(dev_input_masks):
                    
        mask = mask.bool()
        # take only mask==1 portion
        ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
        
        # flatten entity label
        ent_label.append(torch.masked_select(dev_entity_labels[i], mask))

    
    ent_pred = ent_pred[0].tolist()
    ent_label = ent_label[0].tolist()
    
    # same length as the input
    act_ent_pred = np.array(ent_pred[1:-1])[nndev_data['valid_starts'][:-1]].tolist()
    
    outdir_html = "experiments/test_single/"
    if not os.path.exists(outdir_html):
        os.mkdir(outdir_html)
    
    single_test_fn = os.path.join(outdir_html + "single_test_" + ''.join(nndev_data['words'][:2]) + ".html")
    single_test_fid = open(single_test_fn,'w')
    
    new_to_write = ""
    for _indx, _label in enumerate(act_ent_pred):
        if _label == protein_id or _label == entity_id:
            new_to_write = new_to_write + ' ' + '<span style="color: red">' + \
                           nndev_data['words'][_indx] + '</span>'
            act_ent_pred[_indx] = 1
        else:
            new_to_write = new_to_write + ' ' +nndev_data['words'][_indx]
            act_ent_pred[_indx] = 0
        
    single_test_fid.write(new_to_write)
    single_test_fid.close()
    ##########################################

    return act_ent_pred
    
 




def main(args):

    data_dir = args.data_dir

    

    args._label_to_id_t = OrderedDict([('O', 0), ('Negative_regulation', 1), ('Gene_expression', 2), 
                                       ('Regulation', 3), ('Transcription', 4), ('Positive_regulation', 5), 
                                       ('Binding', 6), ('Localization', 7), ('Phosphorylation', 8), 
                                       ('Protein_catabolism', 9), ('Protein', 10), ('Entity',11)])
    
    args._id_to_label_t = {0: 'O', 1: 'Negative_regulation', 2: 'Gene_expression', 
                           3: 'Regulation', 4: 'Transcription', 5: 'Positive_regulation', 
                           6: 'Binding', 7: 'Localization', 8: 'Phosphorylation', 
                           9: 'Protein_catabolism', 10: 'Protein', 11:'Entity'}
    
    args._label_to_id_i = {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
    args._id_to_label_i = {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
    args.arg_label_trig = {'trig': 6}
    args.trig_arg_label = {6:'trig'}
    args.SIMPLE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization', 'Phosphorylation']
    args.REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
    args.BIND = ['Binding']

    args._label_to_id_ii = {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5, 'CSite': 6}
    args._id_to_label_ii = {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc', 6: 'CSite'}
    
    
    
    

    if 'scibert' in args.model:
        bert_weights_path = 'scibert_scivocab_uncased'
            
    elif 'biobert' in args.model:
        bert_weights_path= 'biobert_v1.1_pubmed'
    elif 'bert' in args.model:
        bert_weights_path=args.model


    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        if args.no_cuda:
            args.n_gpu = 0
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
        
    args.device = device
    
    
    
    config_class, model_class, tokenizer_class = MODEL_CLASSES['bert']
        
    config = config_class.from_pretrained(
        bert_weights_path,cache_dir = None)
    
    tokenizer = tokenizer_class.from_pretrained(
        bert_weights_path,
        do_lower_case = True, cache_dir = None)
    
    # Set seed
    # set_seed(args)
    
    
    pretrain_bert = PreMultitaskClassBase.from_pretrained(
            bert_weights_path,
            from_tf=bool(".ckpt" in bert_weights_path),
            config=config)
    
    pretrain_bert.resize_token_embeddings(len(tokenizer))
    
    
    
    
    model = BertMultitaskClassifier(args, config, pretrain_bert)
    
    model.to(args.device)
    
    
    model_encode = PreMultitaskClassBase.from_pretrained(
            bert_weights_path,
            from_tf=bool(".ckpt" in bert_weights_path),
            config=config)
    
    model_encode.to(args.device)
    
    
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s",
        args.local_rank,
        device,
        args.n_gpu
    )
    logger.info("Training/evaluation parameters %s", args)
    
    if args.do_test:
        
        model = BertMultitaskClassifier(args, config, bert_weights_path=bert_weights_path)
        
        output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
        print(output_model_file)
        
        checkpoint = torch.load(output_model_file)
        model.load_state_dict(checkpoint)

        model.cuda()
        model.eval()
        gold=False
        y_trues_e, y_preds_e, y_trues_r, y_preds_r, data_out = model.predict(test_dataloader, gold, args, test=True, eval_kg_datas=test_kg_datas)
         


        write_pkl(data_out, test_abs_spans, test_doc_ids, args.out_test_pkl, gold_tri=False)
        unmerge_normalize(args.output_dir.split('/')[-1])

    if args.do_train:
        gold=True
        
        
        config_path = 'experiments/ge11/configs/train-gold-train.yaml'
        
        with open(config_path, 'r') as stream:
            pred_params = utils1._ordered_load(stream)
        
        
        # Fix seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
        random.seed(pred_params['seed'])
        np.random.seed(pred_params['seed'])
        torch.manual_seed(pred_params['seed'])
        
        with open(pred_params['saved_params'], "rb") as f:
            parameters = pickle.load(f)
        
        # Set predict settings value for params
        parameters['gpu'] = pred_params['gpu']
        parameters['batchsize'] = args.per_gpu_train_batch_size
        
        parameters['train_data'] = pred_params['train_data']
        parameters['dev_data'] = pred_params['dev_data']
        
        parameters['bert_model'] = pred_params['bert_model']
        result_dir = pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        parameters['result_dir'] = pred_params['result_dir']
        
        # raw text
        parameters['raw_text'] = pred_params['raw_text']
        parameters['ner_predict_all'] = pred_params['ner_predict_all']
        parameters['a2_entities'] = pred_params['a2_entities']
        
        # update the argument tags and ids
        parameters['mappings']['nn_mapping']['id_arg_mapping'] = args._label_to_id_i #  {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
        parameters['mappings']['nn_mapping']['arg_id_mapping'] = args._id_to_label_i  # {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
        parameters['mappings']['nn_mapping']['id_arg_trig_mapping'] = args.arg_label_trig # {'trig': 6}
        parameters['mappings']['nn_mapping']['arg_trig_id_mapping'] = args.trig_arg_label# {6:'trig'}
        
        parameters['max_seq_length'] = config.max_position_embeddings
        
        
        
        # train dataset
        
        
        path2 = "./data/GE11_train_candidate_25_positive.pkl"
        f = open(path2,"rb")
        ge11_train_candi_data = pickle.load(f)
        f.close()
        
        
        train_data = prepdata.prep_input_data(pred_params['train_data'], parameters) 
        nntrain_data, train_dataloader,train_data_ = read_test_data(train_data,ge11_train_candi_data,parameters, args)
        
        
        
        
        # dev dataset
        
        dev_data = prepdata.prep_input_data(pred_params['dev_data'], parameters) 
        
        path3 = "./data/GE11_dev_candidate_25_positive.pkl"
        f = open(path3,"rb")
        ge11_dev_candi_data = pickle.load(f)
        f.close()
        
        
        nndev_data, dev_dataloader, dev_data_ = read_test_data(dev_data,ge11_dev_candi_data,parameters,args)
        
        
        ## load the state of art model to finetune
        if args.use_SOTA_model:
            checkpoints = [args.output_dir]
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
            logger.info("Evaluate the following checkpoints: %s", checkpoints)
            num_chek = 0
            for checkpoint in checkpoints:
                num_chek += 1
                global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
                prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
                model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
                model.to(args.device)
                print("load the SOTA model successfully!\n")
        
        model.to(args.device)
        # args.per_gpu_train_batch_size = parameters['batchsize']
        
        
        loss_train_tt, loss_eval_tt = _training(model,dev_dataloader,nndev_data, dev_data_,
                                                dev_dataloader,nndev_data, dev_data_,
                                                gold, args,tokenizer, model_encode)
        
        
        
        print(loss_train_tt)
        print(loss_eval_tt)

        a_file = open("EvntExtraction_fullGe11_train_eva_addcandi_20230815_2.json", "w")
        jj = json.dumps({'train':loss_train_tt, 'eval':loss_eval_tt})
        a_file.write(jj)
        a_file.close()
        
    if args.do_eval: # add the external knowledge information
        gold=True
        
        config_path = 'experiments/ge11/configs/train-gold-train.yaml'
        
        with open(config_path, 'r') as stream:
            pred_params = utils1._ordered_load(stream)
        
        
        # Fix seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
        random.seed(pred_params['seed'])
        np.random.seed(pred_params['seed'])
        torch.manual_seed(pred_params['seed'])
        
        with open(pred_params['saved_params'], "rb") as f:
            parameters = pickle.load(f)
        
        # Set predict settings value for params
        parameters['gpu'] = pred_params['gpu']
        parameters['batchsize'] = 1
        
        
        # Set evaluation settings
        parameters['train_data'] = pred_params['train_data']
        parameters['dev_data'] = pred_params['dev_data']
        
        parameters['bert_model'] = pred_params['bert_model']
        result_dir = pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        parameters['result_dir'] = pred_params['result_dir']
        
        # raw text
        parameters['raw_text'] = pred_params['raw_text']
        parameters['ner_predict_all'] = pred_params['ner_predict_all']
        parameters['a2_entities'] = pred_params['a2_entities']
        
        # update the argument tags and ids
        parameters['mappings']['nn_mapping']['id_arg_mapping'] = args._label_to_id_i #  {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
        parameters['mappings']['nn_mapping']['arg_id_mapping'] = args._id_to_label_i  # {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
        parameters['mappings']['nn_mapping']['id_arg_trig_mapping'] = args.arg_label_trig # {'trig': 6}
        parameters['mappings']['nn_mapping']['arg_trig_id_mapping'] = args.trig_arg_label# {6:'trig'}
        
        parameters['max_seq_length'] = config.max_position_embeddings
        
        dev_data = prepdata.prep_input_data(pred_params['dev_data'], parameters) 
        
        
        path3 = "./data/GE11_dev_candidate_25_positive.pkl"
        f = open(path3,"rb")
        ge11_dev_candi_data = pickle.load(f)
        f.close()    
        
        nndev_data, dev_dataloader, dev_data_ = read_test_data(dev_data, ge11_dev_candi_data, parameters,args)
        
        checkpoints = [args.output_dir]
        
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)
            result, dev_metric = evaluate(model, dev_dataloader,nndev_data, dev_data_,
                                        gold, args,tokenizer, model_encode, prefix=prefix)
            
            print(result)
            print(dev_metric)
            
            f = open("gena11_relation_entity.json","w")
            jj = json.dumps(dev_metric)
            f.write(jj)
            f.close()



    if args.do_test_single:
        gold=True
        
        config_path = 'experiments/ge11/configs/train-gold-train.yaml'
        
        with open(config_path, 'r') as stream:
            pred_params = utils1._ordered_load(stream)
        
        
        # Fix seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
        random.seed(pred_params['seed'])
        np.random.seed(pred_params['seed'])
        torch.manual_seed(pred_params['seed'])
        
        with open(pred_params['saved_params'], "rb") as f:
            parameters = pickle.load(f)
        
        # Set predict settings value for params
        parameters['gpu'] = pred_params['gpu']
        parameters['batchsize'] = 1
        
        
        # Set evaluation settings
        parameters['train_data'] = pred_params['train_data']
        parameters['dev_data'] = pred_params['dev_data']
        
        parameters['bert_model'] = pred_params['bert_model']
        result_dir = pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        parameters['result_dir'] = pred_params['result_dir']
        
        # raw text
        parameters['raw_text'] = pred_params['raw_text']
        parameters['ner_predict_all'] = pred_params['ner_predict_all']
        parameters['a2_entities'] = pred_params['a2_entities']
        
        # update the argument tags and ids
        parameters['mappings']['nn_mapping']['id_arg_mapping'] = args._label_to_id_i #  {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
        parameters['mappings']['nn_mapping']['arg_id_mapping'] = args._id_to_label_i  # {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
        parameters['mappings']['nn_mapping']['id_arg_trig_mapping'] = args.arg_label_trig # {'trig': 6}
        parameters['mappings']['nn_mapping']['arg_trig_id_mapping'] = args.trig_arg_label# {6:'trig'}
        
        parameters['max_seq_length'] = config.max_position_embeddings
        
        
        
        from loader.sentence import prep_sentence_offsets
        
        test_sentence = 'A potential role of BMP - 6 in the immune system has been implied by various studies of malignant and rheumatoid diseases'
        test_input = {'0':[test_sentence]}
    
        test_sentence1 = prep_sentence_offsets(test_input)
    
        
        
        # get the subwords information
        words = test_sentence1['doc_data']['0'][0]['words']
        subwords = []
        subword_offset_mapping = {}
        subword_pos = 0
        valid_starts = [0]
        
        for token_idx, token in enumerate(words):
            subtokens = tokenizer.tokenize(token)
            if subtokens:
                subword_offset_mapping[subword_pos] = token_idx
                subword_pos += 1
                subwords.append(subtokens[:1][0])
                for subtoken in subtokens[1:]:
                    subword_offset_mapping[subword_pos] = token_idx
                    subword_pos += 1
                    subwords.append(subtoken)
                valid_starts.append(len(subwords))
                    
        # prepare the input for Bert model            
        from tokenization_bert import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('biobert_v1.1_pubmed', do_lower_case = False)
        
        num_tokens = len(subwords)
        token_mask = [1]*num_tokens
        
        ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + subwords + ["[SEP]"])
        token_mask = [0] + token_mask + [0]
        num_att = len(token_mask)
        att_mask = [1]*num_att
        
            # add padding
        max_seq_length = parameters['max_seq_length'] 
        pad_len = max_seq_length - num_att
        
        ids += [0]*pad_len
        token_mask += [0]*pad_len
        att_mask += [0]*pad_len
        
        # test single data 
        nntest_data = { 'ids': ids, 'attention_mask': att_mask, 'token_mask': token_mask,
                        'words': words,
                        'sub_words': subwords,
                        'offsets': test_sentence1['doc_data']['0'][0]['offsets'],
                        'sentence': test_sentence,
                        'valid_starts':valid_starts  
        }
        
        
        
        checkpoints = [args.output_dir]
        
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)
            result = predict_single(model, None, nntest_data, None,
                                        gold, args,tokenizer, model_encode, prefix=prefix)
            
            print(result)
            
            
            f = open("test_entity.json","w")
            jj = json.dumps(result)
            f.write(jj)
            f.close()

        
        print()
    
    
    
    if args.single_test_file:
        gold=True
        
        config_path = 'experiments/ge11/configs/train-gold-train.yaml'
        
        with open(config_path, 'r') as stream:
            pred_params = utils1._ordered_load(stream)
        
        
        # Fix seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
        random.seed(pred_params['seed'])
        np.random.seed(pred_params['seed'])
        torch.manual_seed(pred_params['seed'])
        
        with open(pred_params['saved_params'], "rb") as f:
            parameters = pickle.load(f)
        
        # Set predict settings value for params
        parameters['gpu'] = pred_params['gpu']
        parameters['batchsize'] = 1
        
        
        # Set evaluation settings
        parameters['train_data'] = pred_params['train_data']
        parameters['dev_data'] = pred_params['dev_data']
        
        parameters['bert_model'] = pred_params['bert_model']
        result_dir = pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        parameters['result_dir'] = pred_params['result_dir']
        
        # raw text
        parameters['raw_text'] = pred_params['raw_text']
        parameters['ner_predict_all'] = pred_params['ner_predict_all']
        parameters['a2_entities'] = pred_params['a2_entities']
        
        # update the argument tags and ids
        parameters['mappings']['nn_mapping']['id_arg_mapping'] = args._label_to_id_i #  {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
        parameters['mappings']['nn_mapping']['arg_id_mapping'] = args._id_to_label_i  # {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
        parameters['mappings']['nn_mapping']['id_arg_trig_mapping'] = args.arg_label_trig # {'trig': 6}
        parameters['mappings']['nn_mapping']['arg_trig_id_mapping'] = args.trig_arg_label# {6:'trig'}
        
        parameters['max_seq_length'] = config.max_position_embeddings
        
        
        
        from loader.sentence import prep_sentence_offsets
        
        
        checkpoints = [args.output_dir]
        
        # checkpoints = list(
        #     os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        # )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)
            
            
            test_file_dir = args.test_input_file
    
            outdir_html = "experiments/test_single/"
            if not os.path.exists(outdir_html):
                os.mkdir(outdir_html)
            
            single_test_fn = os.path.join(outdir_html + "single_test_" + ''.join(test_file_dir[:4]) + ".html")
            single_test_fid = open(single_test_fn,'w')
            
            file_seed = open(test_file_dir,'r')
            
            for line in file_seed:
                print(line)
            
            
                test_sentence = line
                test_input = {'0':[test_sentence]}
            
                test_sentence1 = prep_sentence_offsets(test_input)
                
                
                # get the subwords information
                words = test_sentence1['doc_data']['0'][0]['words']
                subwords = []
                subword_offset_mapping = {}
                subword_pos = 0
                valid_starts = [0]
                
                for token_idx, token in enumerate(words):
                    subtokens = tokenizer.tokenize(token)
                    if subtokens:
                        subword_offset_mapping[subword_pos] = token_idx
                        subword_pos += 1
                        subwords.append(subtokens[:1][0])
                        for subtoken in subtokens[1:]:
                            subword_offset_mapping[subword_pos] = token_idx
                            subword_pos += 1
                            subwords.append(subtoken)
                        valid_starts.append(len(subwords))
                            
                # prepare the input for Bert model            
                from tokenization_bert import BertTokenizer
                tokenizer = BertTokenizer.from_pretrained('biobert_v1.1_pubmed', do_lower_case = False)
                
                num_tokens = len(subwords)
                token_mask = [1]*num_tokens
                
                ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + subwords + ["[SEP]"])
                token_mask = [0] + token_mask + [0]
                num_att = len(token_mask)
                att_mask = [1]*num_att
                
                    # add padding
                max_seq_length = parameters['max_seq_length'] 
                pad_len = max_seq_length - num_att
                
                ids += [0]*pad_len
                token_mask += [0]*pad_len
                att_mask += [0]*pad_len
                
                # test single data 
                nntest_data = { 'ids': ids, 'attention_mask': att_mask, 'token_mask': token_mask,
                                'words': words,
                                'sub_words': subwords,
                                'offsets': test_sentence1['doc_data']['0'][0]['offsets'],
                                'sentence': test_sentence,
                                'valid_starts':valid_starts  
                }
                
                act_ent_pred = predict_single(model, None, nntest_data, None,
                                            gold, args,tokenizer, model_encode, prefix=prefix)
            
                new_to_write = "<br>"
                protein_id = args._label_to_id_t['Protein'] 
                entity_id = args._label_to_id_t['Entity']
                
                for _indx, _label in enumerate(act_ent_pred):
                    if _label == protein_id or _label == entity_id:
                        new_to_write = new_to_write + ' ' + '<span style="color: red">' + \
                                    nntest_data['words'][_indx] + '</span>'
                        act_ent_pred[_indx] = 1
                    else:
                        new_to_write = new_to_write + ' ' +nntest_data['words'][_indx]
                        act_ent_pred[_indx] = 0
            
                new_to_write = new_to_write + '<br>'    
                single_test_fid.write(new_to_write)
            
                print(act_ent_pred)
                print(nntest_data['words'])
            
            single_test_fid.write(new_to_write)
            single_test_fid.close()
            # f = open("test_entity.json","w")
            # jj = json.dumps(result)
            # f.write(jj)
            # f.close()
    
    
    
    
    
    
    if args.do_test_ELdata:
        gold=True
        
        config_path = 'experiments/ge11/configs/train-gold-train.yaml'
        
        with open(config_path, 'r') as stream:
            pred_params = utils1._ordered_load(stream)
        
        
        # Fix seed for reproducibility
        os.environ["PYTHONHASHSEED"] = str(pred_params['seed'])
        random.seed(pred_params['seed'])
        np.random.seed(pred_params['seed'])
        torch.manual_seed(pred_params['seed'])
        
        with open(pred_params['saved_params'], "rb") as f:
            parameters = pickle.load(f)
        
        # Set predict settings value for params
        parameters['gpu'] = pred_params['gpu']
        parameters['batchsize'] = 1
        
        
        parameters['bert_model'] = pred_params['bert_model']
        result_dir = pred_params['result_dir']
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        parameters['result_dir'] = pred_params['result_dir']
        
        # raw text
        parameters['raw_text'] = pred_params['raw_text']
        parameters['ner_predict_all'] = pred_params['ner_predict_all']
        parameters['a2_entities'] = pred_params['a2_entities']
        
        # update the argument tags and ids
        parameters['mappings']['nn_mapping']['id_arg_mapping'] = args._label_to_id_i #  {'O': 0, 'Theme': 1, 'Cause': 2, 'Site':3, 'ToLoc': 4, 'AtLoc': 5}
        parameters['mappings']['nn_mapping']['arg_id_mapping'] = args._id_to_label_i  # {0: 'O', 1: 'Theme', 2: 'Cause', 3:'Site', 4:'ToLoc', 5: 'AtLoc'}
        parameters['mappings']['nn_mapping']['id_arg_trig_mapping'] = args.arg_label_trig # {'trig': 6}
        parameters['mappings']['nn_mapping']['arg_trig_id_mapping'] = args.trig_arg_label# {6:'trig'}
        
        parameters['max_seq_length'] = config.max_position_embeddings
        
        
        
        
        path4 = "./data/BC4GE_data_PosiNegaCandi_dev25.json"
        a_file = open(path4, "r")
        Gene_data_PosiNega = json.loads(a_file.read())
        a_file.close()
        
        
        nndev_data, dev_data, all_seq_tags,all_result,all_protname \
        = prepdata.load_and_creat_BC_datasets(args, parameters, tokenizer, Gene_data_PosiNega)
        
        
        test_data_size = len(nndev_data['nn_data']['ids'])

        test_data_ids = TensorDataset(torch.arange(test_data_size))
        test_sampler = SequentialSampler(test_data_ids)
        test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=parameters['batchsize'])
    
        
        
        checkpoints = [args.output_dir]
        
        # checkpoints = list(
        #     os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        # )
        logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        num_chek = 0
        for checkpoint in checkpoints:
            num_chek += 1
            global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
            model.load_state_dict(torch.load(os.path.join(checkpoint, 'pytorch_model-1000000.bin')))
            model.to(args.device)
            result = predict(model, test_dataloader,nndev_data, dev_data,
                                        gold, args,tokenizer, model_encode, prefix=prefix)
            
            a_file = open("EvntExtraction_for_EL_dev.json", "w")
            jj = json.dumps(result)
            a_file.write(jj)
            a_file.close()

def read_test_data(test_data, test_candi, params, args):
    test = prep4nn.data2network(test_data, 'train', params) # need to assign the arguements for each sentence
    # data2network--> entity2network--> ent2net.entity2network 
    # -->[1. entity.convert_to_sub_words; 2.entity.extract_entities]
    
    # add candidate data here!
    
    
    candi_data, candi_scores, candi_data_masks = prep4nn.candi2network(test_candi,params)
    
    
    if len(test) == 0:
        raise ValueError("Test set empty.")
    
    test_data_batch_ = prep4nn.torch_data_2_network(cdata2network=test, params=params, args = args, do_get_nn_data=True)
    
    test_data_batch = prep4nn.add_candiInfo(test_data_batch_,candi_data, candi_scores, candi_data_masks)
    
    
    te_data_size = len(test_data_batch['nn_data']['ids'])

    test_data_ids = TensorDataset(torch.arange(te_data_size))
    test_sampler = SequentialSampler(test_data_ids)
    test_dataloader = DataLoader(test_data_ids, sampler=test_sampler, batch_size=params['batchsize'])
    return test_data_batch, test_dataloader, test 



if __name__ == '__main__':
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str, default = './preprocessed_data/')
    p.add_argument('-other_dir', type=str, default = '../other')
    # select model
    p.add_argument('-model', type=str, default='scibert')#, 'multitask/gold', 'multitask/pipeline'
    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=100)
    p.add_argument('-num_layers', type=int, default=1)
    
    # p.add_argument("-train_batch_size",
    #                     default=1,
    #                     type=int,
    #                     help="Total batch size for training.")
    
    p.add_argument('-per_gpu_train_batch_size', 
                        default = 1,
                        type = int,
                        help = 'total size per gpu for training')
    p.add_argument("-eval_batch_size",
                        default=1,
                        type=int,
                        help="Total batch size for eval.")
    
    
    p.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    p.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    p.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    p.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    p.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    

                
    # p.add_argument('-data_type', type=str, default="matres")
    
    
    
    p.add_argument('-pipe_epoch', type=int, default=100) # 1000: no pipeline training; otherwise <= epochs
    # p.add_argument('-seed', type=int, default=123)
    # p.add_argument('-lr', type=float, default=1e-5) # default 3e-5
    
    p.add_argument('-num_classes', type=int, default=2) # get updated in main()
    p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-relation_weight', type=float, default=1.0)
    p.add_argument('-entity_weight', type=float, default=1.0)
    p.add_argument('-load_model', type=str2bool, default=False)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-fine_tune', type=bool, default=False)
    p.add_argument('-eval_gold',type=bool, default=True)
    # new_add
    p.add_argument('-opt', choices=['adam', 'sgd', 'adagrad'], default='adagrad')
    p.add_argument('-exp_id', type=str, default='test')
    p.add_argument('-random_seed', type=int, default=123)
    
    
    p.add_argument('-do_eval', type=str2bool, default=False, help='load trained models and predict to write pkls, \
                                                                                Note this is different from args.load_model. This is used only when you have the trained models \
                                                                                and want to write out the pkl')
    p.add_argument('-do_train', type=str2bool, default=False, help='true to train the model')                                                                                
    p.add_argument('-do_test', type=str2bool, default=False, help='true to predict on the test set')                                                                                    
    p.add_argument('-do_test_ELdata', type=str2bool, default=False, help='true to predict on the BC4GO set')                                                                                    
    
    p.add_argument('-do_test_single', type=str2bool, default=False, help='true to predict on the test single sentence') 
    
    p.add_argument("--single_test_file", action="store_true", help="Whether to do single test from file.")
    
    p.add_argument("--test_input_file", type=str, default = 'test1.txt')
    
    p.add_argument('-gradient_accumulation_steps',
                        type=int,
                        default=16,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    p.add_argument('-epochs', type=int, default=4)
    
    
    
    p.add_argument('-add_candi',type = str2bool, default=True)
    p.add_argument('-use_SOTA_model', type = str2bool, default = False)
    p.add_argument('-output_dir', type=str, default='local_models/')
    
    p.add_argument('-cuda', type=str2bool, default=True)
    p.add_argument('-n_gpu', type=str, default=1)
    p.add_argument('-no_cuda', action="store_true")
    p.add_argument('-gnn_type', type=str,default = 'ECGAT')
    p.add_argument("-local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")



    p.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    args = p.parse_args()
    args.save_stamp = "%s_hid%s_dropout%s_ew%s" % (args.save_stamp, args.hid, args.dropout, args.entity_weight)

    # detertmine the prefix for processed data based on the model name
    if 'biobert_large' in args.model:
        prefix = 'GE11_biobert_large'
    elif 'biobert' in args.model:
        prefix = 'GE11_biobert_v1.1_pubmed'
    elif 'scibert' in args.model:
        prefix = 'GE11_scibert_scivocab_uncased'
    elif 'bert' in args.model:
        prefix = 'GE11_bert-base-uncased'
    else:
        raise NotImplementedError

    args.SIMPLE = ['Gene_expression', 'Transcription', 'Protein_catabolism', 'Localization', 'Phosphorylation']
    args.REG = ['Negative_regulation', 'Positive_regulation', 'Regulation']
    args.BIND = ['Binding']

    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.n_gpu)

    print(f"Output Directory {args.output_dir}")

    
    main(args)

