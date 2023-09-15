"""Prepare data with span-based for training networks."""

import numpy as np
import itertools
from collections import namedtuple
from collections import defaultdict


Term = namedtuple('Term', ['id2term', 'term2id', 'id2label'])

def get_span_index(
        span_start,
        span_end,
        max_span_width,
        max_sentence_length,
        index,
        limit
):
    assert span_start <= span_end
    assert index >= 0 and index < limit
    assert max_span_width > 0
    assert max_sentence_length > 0

    max_span_width = min(max_span_width, max_sentence_length)
    invalid_cases = max(
        0, span_start + max_span_width - max_sentence_length - 1
    )
    span_index = (
            (max_span_width - 1) * span_start
            + span_end
            - invalid_cases * (invalid_cases + 1) // 2
    )
    return span_index * limit + index


def get_batch_data(fid, entities, \
                   terms, valid_starts, sw_sentence, sw_sentences_arg, term_ev, readable_ent, \
                   tokenizer, params, args):
    mlb = params["mappings"]["nn_mapping"]["mlb"]

    max_entity_width = params["max_entity_width"]
    max_trigger_width = params["max_trigger_width"]
    max_span_width = params["max_span_width"]
    max_seq_length = params["max_seq_length"]
    
    

    
    
    tokens = [token for token, *_ in sw_sentence]
    
    
    
    num_tokens = len(tokens)
    token_mask = [1] * num_tokens
    
    
    ids = tokenizer.convert_tokens_to_ids(["[CLS]"] + tokens + ["[SEP]"])
    token_mask = [0] + token_mask + [0]
    
    num_att = len(token_mask)
    att_mask = [1]*num_att    
   

    

    # Account for [CLS] and [SEP] tokens
    if num_att > max_seq_length:
        num_tokens = max_seq_length -1
        tokens = tokens[:max_seq_length]
        ids = ids[:max_seq_length]
        token_mask = token_mask[:max_seq_length]
        att_mask = att_mask[:max_seq_length]
    else:
        pad_len = max_seq_length - num_att
        
        ids += [0]*pad_len
        token_mask += [0]*pad_len
        att_mask += [0]*pad_len        
        
        
        

    # ! Whether use value 1 for [CLS] and [SEP]
    # attention_mask = [1] * len(ids)

    
    
    # get all entity lables and ids
    ent_labels_name = [_enlabel for _, _enlabel, *_ in sw_sentence]
    ent_labels = []
    for _enlabel in ent_labels_name:
        if _enlabel[0].startswith('B') or _enlabel[0].startswith('I'):
            _enlabel_id = params['mappings']['nn_mapping']['tag_id_mapping'][_enlabel[0][2:]]
        else:
            _enlabel_id = params['mappings']['nn_mapping']['tag_id_mapping'][_enlabel[0]]
        ent_labels.append(_enlabel_id)
        
    ent_labels = [0] + ent_labels + [0] # pad the [CLS] and [SEP]
    _num_label = len(ent_labels)
    
    protein_id =  params['mappings']['nn_mapping']['tag_id_mapping']['Protein']
    entity_id = params['mappings']['nn_mapping']['tag_id_mapping']['Entity']
    ent_labels_np = np.array(ent_labels)
    
    prot_idx = np.where(ent_labels_np == protein_id)[0].tolist() # [5 6 7 8 9 10]
    enti_idx = np.where(ent_labels_np == entity_id)[0].tolist() 
    
    prot_idx = prot_idx + enti_idx
    
    
    tri_idx = np.where((ent_labels_np> 0) *( ent_labels_np != protein_id)*(ent_labels_np != entity_id))[0].tolist()
    
    # trigger entity pairs
    te_pairs = list(itertools.product(tri_idx, prot_idx))
    # trigger trigger pairs
    tt_pairs = [(i, j) for i in tri_idx for j in tri_idx if i != j and args._id_to_label_t[ent_labels[i]] in args.REG]

    pred_rels = te_pairs + tt_pairs
    
    
    if _num_label > max_seq_length:
        ent_labels = ent_labels[:max_seq_length]
    else:
        pad_len = max_seq_length - _num_label
        ent_labels += [0]*pad_len    
    
    tri_arg_posipair_goldlabel = defaultdict(list) # get the interaction label for each pair of trigger and arg
    
    tri_arg_posipair_goldlabel_all = defaultdict(list) # get the interaction label for each pair of trigger and arg all
    
    # creat trigger and entity interactions (paired positions) and their arguement labels
    tri_arg_interactions = [] # [ev1,ev2,...] --> ev1 = [trig_posi,entity_posi]
    tri_arg_labels = [] # [ev1,ev2,...] --> ev1 = arguement label*num_posi of trig and arg

    arg_labels = {} # arguement labels by token order
    for evid in sw_sentences_arg:
        arg_labels[evid] = [] # labels for arguments and triggers
        for _, _arglabel, *_ in sw_sentences_arg[evid]:
            
            if _arglabel == 'CSite':
                _arglabel_id = params['mappings']['nn_mapping']['id_arg_mapping']['Site']
                
            elif _arglabel not in params['mappings']['nn_mapping']['id_arg_mapping']: # add trigger labels
                _arglabel_id = params['mappings']['nn_mapping']['id_arg_trig_mapping']['trig']
            else:
                _arglabel_id = params['mappings']['nn_mapping']['id_arg_mapping'][_arglabel]
        
            arg_labels[evid].append(_arglabel_id)

        arg_labels[evid] = [0] + arg_labels[evid] + [0]
        
            # get the position of argument
        arg_posi_temp_all = np.where(np.array(arg_labels[evid])> 0)[0].tolist() # these may contain multiple args
        # for tri_arg_interactions && tri_ent_labels
        tri_arg_inter_each = []
            # get the position of trigger
        tri_posi_temp = [i for i, e in enumerate(arg_labels[evid]) \
                         if e == params['mappings']['nn_mapping']['id_arg_trig_mapping']['trig']] 
        
        if tri_posi_temp == []:
            continue
            for _start_end in terms:
                if terms[_start_end][0].startswith('TR'):
                    tri_posi_temp = [i for i in range(_start_end[0]+1,_start_end[1]+1,1)]

                    
                    
        num_arg_each = len(term_ev[evid][3])
        label_each = []
        
        for i in range(num_arg_each):
            _arg_name = term_ev[evid][4][i] # argument tag/name eg: T1 
            
            # find the correct argument tag/name if it is an event previously, eg TR1 or T1
            while _arg_name.startswith('E'):
                if _arg_name not in term_ev:
                    break
                else:
                    temp_ev_id  = _arg_name
                    _arg_name = term_ev[temp_ev_id][2]
            
            if _arg_name not in readable_ent:
                continue    
            else:
                posi_st = 0
                posi_end = 0
                for star_end in terms:
                    if terms[star_end][0] == _arg_name:
                        posi_st = star_end[0]
                        posi_end = star_end[1]+1
                
                if posi_end == 0: # if from previous terms cannot find the _arg_name
                    posi_st = valid_starts[readable_ent[_arg_name]['toks'][0]]
                    posi_end = valid_starts[readable_ent[_arg_name]['toks'][-1]]+1
                      
                _arg_posi_temp =[posi+1 for posi in range(posi_st,posi_end,1)] # consider the [CLS] tok
            
            if _arg_posi_temp == []:
                 continue
            
            _arglabel_id = arg_labels[evid][_arg_posi_temp[0]]
            
            label_each.append((len(tri_posi_temp)+len(_arg_posi_temp))*[_arglabel_id])

            
            tri_arg_inter_each.append([tri_posi_temp, _arg_posi_temp])
            
            label_each_product = (len(tri_posi_temp)*len(_arg_posi_temp))*[_arglabel_id]
            
            for _tri_posi in tri_posi_temp:
                for _ent_posi in _arg_posi_temp:
                    tri_arg_posipair_goldlabel[(_tri_posi, _ent_posi)].append(_arglabel_id)
            
            
                        
        tri_arg_interactions.append(tri_arg_inter_each)
        tri_arg_labels.append(label_each)    

        
        
        
        
        
        
        # padding for arg_labels
        _num_arg = len(arg_labels[evid])
        if _num_arg > max_seq_length:
            arg_labels = arg_labels[evid][:max_seq_length]
        else:
            pad_len = max_seq_length - _num_arg
            arg_labels[evid] += [0]*pad_len
    
    
    for pair_posi in pred_rels:
        if pair_posi not in tri_arg_posipair_goldlabel:
            tri_arg_posipair_goldlabel_all[pair_posi].append(0) # not relation
        else:
            tri_arg_posipair_goldlabel_all[pair_posi].append(tri_arg_posipair_goldlabel[pair_posi][0])
                
    # Generate spans here
    span_starts = np.tile(
        np.expand_dims(np.arange(num_tokens), 1), (1, max_span_width)
    )  # (num_tokens, max_span_width)

    span_ends = span_starts + np.expand_dims(
        np.arange(max_span_width), 0
    )  # (num_tokens, max_span_width)

    # span_indices = []
    # span_labels = []
    # span_labels_match_rel = []
    # entity_masks = []
    # trigger_masks = []
    # span_terms = Term({}, {}, {})

    # for span_start, span_end in zip(
    #         span_starts.flatten(), span_ends.flatten()
    # ):
    #     if span_start >= 0 and span_end < num_tokens:
    #         span_label = []  # No label
    #         span_term = []
    #         span_label_match_rel = 0

    #         entity_mask = 1
    #         trigger_mask = 1

    #         if span_end - span_start + 1 > max_entity_width:
    #             entity_mask = 0
    #         if span_end - span_start + 1 > max_trigger_width:
    #             trigger_mask = 0

    #         # Ignore spans containing incomplete words
    #         valid_span = True
    #         if not (params['predict'] and (params['pipelines'] and params['pipe_flag'] != 0)):
    #             if span_start not in valid_starts or (span_end + 1) not in valid_starts:
    #                 # Ensure that there is no entity label here
    #                 if not (params['predict'] and (params['pipelines'] and params['pipe_flag'] != 0)):
    #                     assert (span_start, span_end) not in entities

    #                     entity_mask = 0
    #                     trigger_mask = 0
    #                     valid_span = False

    #         if valid_span:
    #             if (span_start, span_end) in entities:
    #                 span_label = entities[(span_start, span_end)]
    #                 span_term = terms[(span_start, span_end)]

    #         if len(span_label) > params["ner_label_limit"]:
    #             print('over limit span_label', span_term)

    #         # For multiple labels
    #         for idx, (_, term_id) in enumerate(
    #                 sorted(zip(span_label, span_term), reverse=True)[:params["ner_label_limit"]]):
    #             span_index = get_span_index(span_start, span_end, max_span_width, num_tokens, idx,
    #                                         params["ner_label_limit"])
    #             span_terms.id2term[span_index] = term_id
    #             span_terms.term2id[term_id] = span_index

    #             # add entity type
    #             term_label = params['mappings']['nn_mapping']['id_tag_mapping'][span_label[0]]
    #             span_terms.id2label[span_index] = term_label

    #         span_label = mlb.transform([span_label])[-1]

    #         span_indices += [(span_start, span_end)] * params["ner_label_limit"]
    #         span_labels.append(span_label)
    #         span_labels_match_rel.append(span_label_match_rel)
    #         entity_masks.append(entity_mask)
    #         trigger_masks.append(trigger_mask)

    return {
        'tokens': tokens,
        'ids': ids,
        'token_mask': token_mask,
        'attention_mask': att_mask,
        'entity_labels': ent_labels,
        'arg_labels': arg_labels,
        'trig_arg_relations': tri_arg_interactions,
        'trig_arg_relation_labels': tri_arg_labels,
        "trig_arg_interaction":pred_rels,
        "trig_arg_interaction_labels":tri_arg_posipair_goldlabel_all
        # 'span_indices': span_indices,
        # 'span_labels': span_labels,
        # 'span_labels_match_rel': span_labels_match_rel,
        # 'entity_masks': entity_masks,
        # 'trigger_masks': trigger_masks,
        # 'span_terms': span_terms
    }


def get_nn_data(fids, entitiess, termss, valid_startss, sw_sentences, sw_sentences_arg, terms_evs, readable_entss, tokenizer, params, args):
    samples = []

    for idx, sw_sentence in enumerate(sw_sentences):
        fid = fids[idx]
        entities = entitiess[idx]
        terms = termss[idx]
        valid_starts = valid_startss[idx]
        sw_sentence_arg = sw_sentences_arg[idx]
        
        term_ev = terms_evs[idx]
        readable_ent = readable_entss[idx]
        # if idx == 1533:
        #     print(1533)
        sample = get_batch_data(fid, entities, terms, valid_starts, 
                                sw_sentence, sw_sentence_arg,
                                term_ev, readable_ent,
                                tokenizer,
                                params, args)
        samples.append(sample)

    all_tokens = [sample["tokens"] for sample in samples]

    all_ids = [sample["ids"] for sample in samples]
    all_token_masks = [sample["token_mask"] for sample in samples]
    all_attention_masks = [sample["attention_mask"] for sample in samples]
    all_entity_labels = [sample['entity_labels'] for sample in samples]
    all_arg_labels = [sample['arg_labels'] for sample in samples]
    
    all_tri_arg_relations = [sample['trig_arg_relations'] for sample in samples]
    all_tri_arg_relation_labels = [sample['trig_arg_relation_labels'] for sample in samples]
        
    all_tri_arg_interactions = [sample['trig_arg_interaction'] for sample in samples]
    all_tri_arg_interaction_labels = [sample['trig_arg_interaction_labels'] for sample in samples]
    # all_span_indices = [sample["span_indices"] for sample in samples]
    # all_span_labels = [sample["span_labels"] for sample in samples]
    # all_span_labels_match_rel = [sample["span_labels_match_rel"] for sample in samples]
    # all_entity_masks = [sample["entity_masks"] for sample in samples]
    # all_trigger_masks = [sample["trigger_masks"] for sample in samples]
    # all_span_terms = [sample["span_terms"] for sample in samples]

    return {
        'tokens': all_tokens,
        'ids': all_ids,
        'token_mask': all_token_masks,
        'attention_mask': all_attention_masks,
        'entity_label': all_entity_labels,
        'arg_label': all_arg_labels,
        
        'tri_arg_rela': all_tri_arg_relations,
        'tri_arg_rela_label': all_tri_arg_relation_labels,
        
        'tri_arg_interaction': all_tri_arg_interactions,
        'tri_arg_interaction_label': all_tri_arg_interaction_labels
        # 'span_indices': all_span_indices,
        # 'span_labels': all_span_labels,
        # 'span_labels_match_rel': all_span_labels_match_rel,
        # 'entity_masks': all_entity_masks,
        # 'trigger_masks': all_trigger_masks,
        # 'span_terms': all_span_terms
    }
