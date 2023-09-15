"""Load data from brat format and process for entity"""

from collections import OrderedDict

from loader.brat import brat_loader
from loader.sentence import prep_sentence_offsets, process_input
from loader.entity import process_etypes, process_tags, process_entities


import logging
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import json



logger = logging.getLogger(__name__)




def prep_input_data(files_fold, params):
    # load data from *.ann files
    entities0, sentences0 = brat_loader(files_fold, params)

    # sentence offsets
    sentences1 = prep_sentence_offsets(sentences0)

    # entity
    entities1 = process_etypes(entities0)  # all entity types
    terms0 = process_tags(entities1)  # terms, offset, tags, etypes
    input0 = process_entities(entities1, sentences1, params, files_fold)

    # prepare for training batch data for each sentence
    input1 = process_input(input0)

    for doc_name, doc in sorted(input0.items(), key=lambda x: x[0]):
        entities = set()
        num_entities_per_doc = 0
        for sentence in doc:
            eids = sentence["eids"]
            entities |= set(eids)
            num_entities_per_doc += len(eids)

        full_entities = set(entities1["pmids"][doc_name]["ids"])
        diff = full_entities.difference(entities)
        if diff:
            print(doc_name, sorted(diff, key=lambda _id: int(_id.replace("T", ""))))

    # entity indices
    g_entity_ids_ = OrderedDict()
    for fid, fdata in entities0.items():
        # get max entity id
        eid_ = [eid for eid in fdata['ids'] if not eid.startswith('TR')]
        ids_ = [int(eid.replace('T', '')) for eid in eid_]
        if len(ids_) > 0:
            max_id = max(ids_)
        else:
            max_id = 0
        eid_.append(max_id)
        g_entity_ids_[fid] = eid_

    return {'entities': entities1, 'terms': terms0, 'sentences': sentences1, 'input': input1,
            'g_entity_ids_': g_entity_ids_}



def load_and_creat_BC_datasets(args, parameters, tokenizer, data):
    
    print("Creating BC features from dataset file")
    features = get_BC_examples_new(data,parameters['max_seq_length'],tokenizer, args,parameters)

    all_bc_token = [f.mention_token_ids for f in features]
    all_bc_mask = [f.mention_token_masks for f in features]
    all_go_info = [f.go_info for f in features]
    all_go_info_mask = [f.go_info_mask for f in features]
    all_text_label = [f.text_label for f in features]
    
    
    all_sequence_tages = [f.sequence_tags for f in features]
    all_result = [f.result for f in features]
    all_protname =[f.mention_textname for f in features]

    dataset = {}
    dataset['nn_data'] = {}
    dataset['nn_data']['ids'] = all_bc_token
    dataset['nn_data']['attention_mask'] = all_bc_mask
    dataset['nn_data']['entity_label'] = all_text_label
    
    dataset['nn_data']['candi_info'] = all_go_info
    dataset['nn_data']['candi_mask'] = all_go_info_mask
    
    
    
    
    
    return dataset, (all_bc_token, all_bc_mask, all_go_info, all_go_info_mask, all_text_label,data), all_sequence_tages, all_result, all_protname



def get_BC_examples_new(Gene_data_PosiNega,max_seq_length,
            tokenizer,
            args, parameters):
   

    features = []
    
    for case_id, val in Gene_data_PosiNega.items():
        
        
        
        Genedata = val[0]
        Gene_trueGo = val[1]
        # Gene_posi = val[2]
        # Gene_nega = val[3]
        
        
        
        tokenized_text_, mention_start_markers, mention_end_markers, sequence_tags \
        = get_mentions_tokens(Genedata,tokenizer)
        
        doc_tokens = tokenizer.convert_tokens_to_ids(tokenized_text_)
        seq_tag_ids = convert_tags_to_ids(sequence_tags)
        
        entity_map = parameters['mappings']['nn_mapping']['tag_id_mapping']
        
        text_label = [0]*len(tokenized_text_)
        for i in range(mention_end_markers[0]-mention_start_markers[0]+1):
            text_label[mention_start_markers[0]+i] = entity_map['Entity']
        
        # the gold candi
        for go_id in Gene_trueGo:
            true_go_term = Gene_trueGo[go_id]
            candi_token, candi_seq = get_candi_tokens(true_go_term,tokenizer)
            candi_seq = convert_tags_to_ids(candi_seq)
            
            candi_token = [tokenizer.cls_token] + candi_token + [tokenizer.sep_token]
            candi_token = tokenizer.convert_tokens_to_ids(candi_token)
            
        
        
        
        
        
        if len(doc_tokens) > max_seq_length:
            print(len(doc_tokens))
            
            doc_tokens = doc_tokens[:max_seq_length]
            doc_token_mask = [1] * max_seq_length
            seq_tag_ids = seq_tag_ids[:max_seq_length]
            text_label = text_label[:max_seq_length]
        else:
            mention_len = len(doc_tokens)
            pad_len = max_seq_length - mention_len
            doc_tokens += [tokenizer.pad_token_id] * pad_len
            text_label += [tokenizer.pad_token_id] * pad_len
            
            doc_token_mask = [1] * mention_len + [0] * pad_len
            seq_tag_ids += [-100]*pad_len

        if len(candi_token) > max_seq_length:
            
            candi_token = candi_token[:max_seq_length]
            candi_token_mask = [1] * max_seq_length
        else:
            candi_len = len(candi_token)
            pad_len = max_seq_length - candi_len
            
            candi_token += [tokenizer.pad_token_id]*pad_len
            candi_token_mask = [1] * candi_len + [0] * pad_len
                
        
                
        result = 1.0
                
        features.append(
            InputFeatures1(
                mention_token_ids = doc_tokens, 
                mention_token_masks = doc_token_mask,
                go_info = candi_token,
                go_info_mask = candi_token_mask,
                text_label = text_label,
                sequence_tags = seq_tag_ids, 
                result = result,
                mention_textname = Genedata['gene_name']
            )
            )

            
            
            
        # tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
        # def convert_tags_to_ids(seq_tags):
        #     seq_tag_ids = [-100]  # corresponds to the [CLS] token
        #     for t in seq_tags:
        #         seq_tag_ids.append(tag_to_id_map[t])
        #     seq_tag_ids.append(-100)  # corresponds to the [SEP] token
        #     return seq_tag_ids
    return features



def get_candi_tokens(CandiData, tokenizer):
    
    candi_text = CandiData['def']
    tokenize_text = tokenizer.tokenize(candi_text)
    sequence_tags = []
    for j, token in enumerate(tokenize_text):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    
    return tokenize_text, sequence_tags


def get_mentions_tokens(Genedata,tokenizer):
    
    start_indx = Genedata['start']
    end_indx = Genedata['end']
    context_text = Genedata['text']
    mention_name = context_text[start_indx: end_indx]        
    
    tokenized_text = [tokenizer.cls_token]
    sequence_tags = []
    mention_start_markers = []
    mention_end_markers = []
    
    # tokenize the text before the mention 
    prefix  = context_text[0:start_indx]
    prefix_tokens = tokenizer.tokenize(prefix)        
    tokenized_text += prefix_tokens
    # The sequence tag for prefix tokens is 'O' , 'DNT' --> 'Do Not Tag'
    for j, token in enumerate(prefix_tokens):
        sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    # Add mention start marker to the tokenized text
    mention_start_markers.append(len(tokenized_text))
    # Tokenize the mention and add it to the tokenized text
    mention_tokens = tokenizer.tokenize(mention_name)
    tokenized_text += mention_tokens
    # Sequence tags for mention tokens -- first token B, other tokens I
    for j, token in enumerate(mention_tokens):
        if j == 0:
            sequence_tags.append('B')
        else:
            sequence_tags.append('I' if not token.startswith('##') else 'DNT')
    # Add mention end marker to the tokenized text
    mention_end_markers.append(len(tokenized_text) - 1)
    
    # text after the mention
    suffix = context_text[end_indx:]
    if len(suffix)>0:
        suffix_tokens = tokenizer.tokenize(suffix)
        tokenized_text += suffix_tokens
        # The sequence tag for suffix tokens is 'O'
        for j, token in enumerate(suffix_tokens):
            sequence_tags.append('O' if not token.startswith('##') else 'DNT')
    tokenized_text += [tokenizer.sep_token]
    
    return tokenized_text, mention_start_markers, mention_end_markers, sequence_tags


tag_to_id_map = {'O': 0, 'B': 1, 'I': 2, 'DNT': -100}
def convert_tags_to_ids(seq_tags):
    seq_tag_ids = [-100]  # corresponds to the [CLS] token
    for t in seq_tags:
        seq_tag_ids.append(tag_to_id_map[t])
    seq_tag_ids.append(-100)  # corresponds to the [SEP] token
    return seq_tag_ids


class InputFeatures1(object):
    def __init__(self, mention_token_ids, mention_token_masks,
                        go_info,go_info_mask,text_label,
                        sequence_tags, result,
                        mention_textname
                 ):
        self.mention_token_ids = mention_token_ids
        self.mention_token_masks = mention_token_masks
        
        self.go_info = go_info
        self.go_info_mask = go_info_mask
        
        self.text_label = text_label
        self.sequence_tags = sequence_tags
        self.result = result
        self.mention_textname = mention_textname

