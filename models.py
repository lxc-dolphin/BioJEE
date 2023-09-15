from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
from utils import *
# from transformers import *
import itertools

from optimization import *
from sklearn.metrics import f1_score, precision_recall_fscore_support
from tqdm import tqdm 
from utils import timer
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import copy

from modeling_bert import BertIntermediate, BertOutput, BertLayer, BertSelfOutput, BertLayer
from modeling_bert import BertModel, BertPreTrainedModel

def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(math.pi / 2) * (x + 0.044715 * x ** 3)))


from collections import defaultdict
try:
    from torch_geometric.nn import GraphConv, TopKPooling, GatedGraphConv, SAGEConv, NNConv, GCNConv, GraphUNet, GATConv
except:
    pass






BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
}



class PreMultitaskClassBase(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        

class BertMultitaskClassifier(BertPreTrainedModel):
    
    def __init__(self, args, config, pretrain_bert):
        super().__init__(config)     
        
        
        self.bert_model = pretrain_bert
        # kg_embedding_dim = args.kg_embedding_dim
        self.config = config
        self.hid_size = args.hid
        
        
        self.num_classes = max(args._label_to_id_i.values()) + 1
        self.num_ent_classes = max(args._label_to_id_t.values()) + 1

        self.dropout = nn.Dropout(p=args.dropout)
        
        # MLP classifier for relation
        self.linear1 = nn.Linear(config.hidden_size *2, args.rel_linear_size)
        self.linear2 = nn.Linear(args.rel_linear_size, self.num_classes)

        # MLP classifier for entity
        self.linear1_ent = nn.Linear(config.hidden_size *2, args.ent_linear_size)
        self.linear2_ent = nn.Linear(args.ent_linear_size, self.num_ent_classes)
          
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_ent = nn.Softmax(dim=2)


    def forward(self, input_ids, entity_labels,
                attention_mask, entity_ids, token_type_ids=None, 
                position_ids=None, head_mask=None,
                arg_rela = None, arg_rela_label = None, 
                rel_idxs=[], lidx=[], ridx=[], task='relation', args=None):
        '''
        entity_labels are just for extracting proteins
        
        '''
        
        
        out = self.bert_model.bert(input_ids, attention_mask,
                                    token_type_ids=token_type_ids,
                                    position_ids=position_ids,
                                    head_mask=head_mask)
        
        # add the entity info to the out 
        out = self.dropout(out[0]) + entity_ids
        ### entity prediction - predict each input token
        if task == 'entity':
            
            # gather protein embeddings
            protein_mask =  torch.cat( out.size(2) * [(entity_labels==10).unsqueeze(2)], dim=2)
            protein_embeddings = out * protein_mask
            
            protein_embeddings = F.avg_pool1d(protein_embeddings.permute(0,2,1), kernel_size=protein_embeddings.size(1), count_include_pad=False).permute(0,2,1)
            
            protein_embeddings = torch.cat( out.size(1) * [protein_embeddings], dim=1)

            
            # add the entity info to protein embedding layer
            # protein_embeddings  = protein_embeddings + entity_ids
            
            # concatenate protein embeddings with the whole sequence
            out = torch.cat([out, protein_embeddings], dim=2)
            out_ent = self.linear1_ent(out) # [1,512,768*2]
            out_ent = self.act(out_ent)
            out_ent = self.linear2_ent(out_ent)
            prob_ent = self.softmax_ent(out_ent)

            # mask = torch.cat([attention_mask] * prob_ent.size(2), dim=2)
            
            return out_ent, prob_ent, out

        ### relaiton prediction - flatten hidden vars into a long vector
        if task == 'relation':
            
            
            # out : [2, 114, 200]
            #### not used
            ltar_f = torch.cat([out[b, lidx[b][r], :].unsqueeze(0) for b,r in rel_idxs], dim=0)
            
            rtar_f = torch.cat([out[b, ridx[b][r], :].unsqueeze(0) for b,r in rel_idxs], dim=0)
            ##########
            # rtar_b = torch.cat([out[b, ridx[b][r], self.hid_size:].unsqueeze(0) for b,r in rel_idxs], dim=0)
            
            # out: [12, 401]
            ###### not used
            out = self.dropout(torch.cat((ltar_f, rtar_f), dim=1))
            ################
            # out = torch.cat((out, fts), dim=1)

            # uncommon below one if use "relation" and "relation_label"
            # out = torch.cat([out[0, tri_arg, :].unsqueeze(0) for ev in arg_rela for pair in ev for tri_arg in pair],dim = 1)
            
            
            # linear prediction
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout(out)
            out = self.linear2(out)
            prob = self.softmax(out)
            return out, prob
        
    def construct_relations(self, entity_logits, entity_labels, attention_masks, 
                            interactions, interaction_labels, args, gold=True, test=False, do_eval = False):
        '''
        ent_probs: Predicted entity probabilty [batch, seq, n_ent_class]
        ents: Gold entities [batch, seq, ], just for identiying proteins
        lengths: The length of each packed sequence [batch]
        pairs: golden 
        ints: interactions
        doc: list of sentence_id
        poss: pos tags
        '''

        
        nopred_rels = []

        ## Case 1: only use gold relation
        if gold:
            # pred_rels = rels
            pred_rels = [interactions]

        ## Case 2: use candidate relation predicted by entity model
        else:
            
            def _is_gold(pair_pred, pairs_gold):
                return pair_pred in pairs_gold

            batch_size = entity_logits.size(0)
            
            

            
            # ent_preds = ent_probs.max(dim=2, keepdim=False)[1].tolist()
            predicted_entities = entity_logits.argmax(dim=2)
            
            # protein_mask
            protein_id = args._label_to_id_t['Protein']  
            none_entity_id = args._label_to_id_t['Protein']

            pred_ints = []
            pred_pairs = []
            for i in range(len(predicted_entities)):

                predicted_entity = predicted_entities[i]
                entity_label = entity_labels[i]
                attention_mask = attention_masks[i]

                # if test, then don't get interaction label 
                if not test:
                    interaction = interactions[i]
                    interaction_label = interaction_labels[i]

                predicted_entity = predicted_entity.cpu().numpy()
                # get the position of all gold proteins in the sentences
                gold_prot_idxs = np.where(entity_label.cpu() == protein_id)[0]
                
                
                # get the position of all predicted triggers in the sentences
                tri_idxs = np.where((predicted_entity > 0) *( predicted_entity != 9))[0].tolist()

                
                # trigger entity pairs
                te_pairs = list(itertools.product(tri_idxs, gold_prot_idxs))
                
                
                # trigger trigger pairs
                tt_pairs = [(i, j) for i in tri_idxs for j in tri_idxs if i != j and args._id_to_label_t[predicted_entity[i]] in args.REG]

                
                pred_int = []
                pred_pairs.append(te_pairs+tt_pairs)
                
                if not test:
                    
                    for p in te_pairs+tt_pairs:
                        
                        if _is_gold(p, interaction):
                            
                            pred_int.append(interaction_label[interaction.index(p)])
                        else:
                            # None event
                            pred_int.append(args._label_to_id_i['None'])
                    pred_ints.append(pred_int)

            
            
            pred_pairs = tuple(pred_pairs)
            pred_ints = tuple(pred_ints)

            pred_rels = pred_pairs
            interaction_labels = pred_ints

        rel_idxs, lidx, ridx = [],[],[]
        
        # need use attention mask to deal with the arg labels
            
        assert len(pred_rels) == entity_logits.size(0)
        
        for i, rel in enumerate(pred_rels):
            # if len(rel) == 2:
            #     print(rel)
            rel_idxs.extend([(i, ii) for ii, _ in enumerate(rel)])
            # if rel != []:
            lidx.append([x[0] for x in rel]) # trigger (tokenized)
            ridx.append([x[1] for x in rel]) # arguement (tokenized)

        # if test, don't return relation label
        if test:
            return None, rel_idxs, lidx, ridx


        rels = [x for rel in pred_rels for x in rel]
        if rels == []:
            labels = torch.FloatTensor([])
        else:
            # labels = torch.LongTensor([ilabel for rel in interaction_labels for x in rel for ilabel in x])
            labels = torch.LongTensor([interaction_labels[pair_][0] for pair_ in rels ])

        # pdb.set_trace()
        
        return labels, rel_idxs, lidx, ridx


    def predict(self, dev_dataloader, gold, args, dev_interactions=None, dev_interaction_labels=None, test=False):
        
        self.eval()
        # need to have a warm-start otherwise there could be no event_pred
        # may need to manually pick poch < #, but 0 generally works when ew is large
        
        with torch.no_grad():
            predicted_interactions = []
            predicted_interaction_labels = []
            predicted_entities = []
            all_gold_interactions = []
            all_gold_interaction_labels = []
            all_gold_entities = []
            all_input_ids = []
            all_sample_ids = []

            ent_pred_map, ent_label_map = {}, {}
            rd_pred_map, rd_label_map = {}, {}
            
            y_trues_e, y_preds_e = [], []
            y_trues_r, y_preds_r = [], []
            for step, batch in enumerate(tqdm(dev_dataloader, desc='Prediction')):
            
                if torch.cuda.is_available():
                    # put the variables onto GPU
                    batch = tuple(t.cuda() for t in batch)
                
                dev_input_ids, dev_input_masks, dev_segment_ids, dev_entity_labels, dev_sample_ids = batch
                
                
                # get sample ids only for data_out
                all_sample_ids.extend(dev_sample_ids.cpu().numpy())
                
                # if args.use_knowledge:
                #     kg_datas = [ eval_kg_datas[sample_id] for sample_id in dev_sample_ids]
                # else:
                #     kg_datas = None
                    
                # entity output
                entity_logits, prob_e, _, _, _ = self.forward(dev_input_ids, dev_entity_labels, dev_input_masks, dev_segment_ids, task='entity',args=args)   # out_e and prob_e: [16, 56, 11]

            
                # mask out the prob of the padding with input mask        
                mask = torch.cat( [dev_input_masks.unsqueeze(2)] * (entity_logits.size(2) ),dim=2)
                mask[:,:,0] = 1
                prob_e *= mask
                
                ## everage 

                
                if not test:
                    gold_interactions = [dev_interactions[sample_id] for sample_id in dev_sample_ids]
                    gold_interaction_labels = [dev_interaction_labels[sample_id] for sample_id in dev_sample_ids]
                    
                    all_gold_interactions.extend(gold_interactions)
                    all_gold_interaction_labels.extend([args._id_to_label_i[label] for labels in gold_interaction_labels for label in labels])
                    # construct relation
                    label_r, rel_idxs, lidx, ridx = self.construct_relations(prob_e, dev_entity_labels, dev_input_masks, gold_interactions, gold_interaction_labels, args, gold=gold, test=test)
                else:
                    label_r, rel_idxs, lidx, ridx = self.construct_relations(prob_e, dev_entity_labels, dev_input_masks, None, None, args, gold=gold, test=test)
                
                assert len(lidx) == len(ridx)

                # retrieve the predicted pairs
                pair_lengths = [len(i) for i in lidx]  # num of pairs in each sent in the batch
                for i in range(len(lidx)): # batch size
                    if len(lidx[i]) == 0:
                        predicted_interactions.append([])
                    else:
                        predicted_interactions.append([i for i in zip(lidx[i], ridx[i])])
                
                ### predict relations
                if rel_idxs != []: # predicted relation could be empty --> skip

                    
                    out_r, prob_r, _, _, _ = self.forward(dev_input_ids, 
                                                          dev_entity_labels,
                                                          dev_input_masks, 
                                                          dev_segment_ids, 
                                                          rel_idxs=rel_idxs, lidx=lidx, ridx=ridx, task='relation', args=args)
                    
                    # (batch, )
                    pred_r = prob_r.data.argmax(dim=1).long().view(-1)
                    if not test:
                        assert pred_r.size(0) == label_r.size(0)

                    if args.cuda:
                        prob_r = prob_r.cpu()
                        if not test:
                            label_r = label_r.cpu()
                    
                    pred_r_list = pred_r.tolist()
                    # extend to all predicted relations
                    y_preds_r.extend(pred_r_list)

                    # retrive the ints labels for the predicted pairs
                    cur = 0
                    for i, l in enumerate(pair_lengths):
                        if pair_lengths[i] == 0:
                            predicted_interaction_labels.append([])
                        else:
                            predicted_interaction_labels.append([args._id_to_label_i[x] for x in pred_r_list[cur:cur+l]])
                            cur += l

                else: # no relation predicted

                    y_preds_r.extend([])
                    predicted_interaction_labels.extend([[] for _ in range(len(dev_input_masks))])

                assert len(predicted_interaction_labels[-1]) ==len(predicted_interactions[-1])
                    
                if not test:
                    y_trues_r.extend(label_r.tolist())

                # retrieve and flatten entity prediction for loss calculation
                ent_pred, ent_label, ent_prob, ent_key, ent_pos, ent_input = [], [], [], [], [], []

                # get entities prediction filtered by mask
                for i, mask in enumerate(dev_input_masks):
                    
                    mask = mask.bool()
                    # take only mask==1 portion
                    ent_pred.append(torch.masked_select(prob_e[i].argmax(dim=1), mask))
                    
                    # flatten entity label
                    ent_label.append(torch.masked_select(dev_entity_labels[i], mask))
                    
                    ent_input.append(torch.masked_select(dev_input_ids[i], mask))

                    all_gold_entities.append(ent_label[-1].tolist())
                    predicted_entities.append(ent_pred[-1].tolist())
                    all_input_ids.append(ent_input[-1].tolist())
                    
                ## collect trigger prediction results
                ent_pred = torch.cat(ent_pred, 0)
                ent_label = torch.cat(ent_label, 0)
                
                
                
                assert ent_pred.size() == ent_label.size() 


                y_trues_e.extend(ent_label.tolist())
                y_preds_e.extend(ent_pred.tolist())

                                

            data_out = {'sample_ids':all_sample_ids, 
            'predicted_entities':predicted_entities,  
            'predicted_interactions': predicted_interactions, 
            'predicted_interaction_labels':predicted_interaction_labels,
            'gold_entities': all_gold_entities,
            'gold_interactions': all_gold_interactions,
            'gold_interaction_labels': all_gold_interaction_labels,
            'input_ids':all_input_ids,
            
            

            }

        return y_trues_e, y_preds_e, y_trues_r, y_preds_r, data_out