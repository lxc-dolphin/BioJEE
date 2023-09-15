"""Prepare data for training networks."""

from collections import OrderedDict

from tokenization_bert import BertTokenizer
from sklearn.preprocessing import MultiLabelBinarizer

from loader.prepNN.sent2net import prep_sentences
from loader.prepNN.ent2net import entity2network, _elem2idx
from loader.prepNN.span4nn import get_nn_data


def candi2network(candi,params):
    
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )
    
    all_candis = []
    all_candi_scores= []
    all_candi_masks = []
    
    max_seq_length = params['max_seq_length']
    
    
    for each_case in candi:
        posi_candi = each_case['posi_candi']
        
        each_candi_info = []
        each_candi_scores = []
        each_candi_masks = []
        
        
        for each_candi in posi_candi:
            candi_info_text = each_candi[1]['def']

             
            
            candi_tokens = tokenizer.tokenize(candi_info_text) 
            tokenized_text = [tokenizer.cls_token] + candi_tokens + [tokenizer.sep_token]
            
            candi_token_id = tokenizer.convert_tokens_to_ids(tokenized_text)
            
            if len(candi_token_id) > max_seq_length:
            
                candi_token_id = candi_token_id[:max_seq_length]
                candi_mask = [1] * max_seq_length
            
            else:
                candi_len = len(candi_token_id)
                pad_len = max_seq_length - candi_len
                
                candi_token_id += [tokenizer.pad_token_id]*pad_len
                candi_mask = [1]*candi_len + [0]*pad_len                
            
            each_candi_info.append(candi_token_id)
            each_candi_masks.append(candi_mask)
            
            _score = each_candi[2]
            each_candi_scores.append(_score)
            
        all_candis.append(each_candi_info)
        all_candi_scores.append(each_candi_scores)
        all_candi_masks.append(each_candi_masks)
        
        
    return all_candis, all_candi_scores, all_candi_masks



def add_candiInfo(data_batch, candi_data, candi_score, candi_masks):
    
    
    data_batch['nn_data']['candi_info'] = candi_data
    data_batch['nn_data']['candi_score'] = candi_score
    data_batch['nn_data']['candi_mask'] = candi_masks
    
    return data_batch






def data2network(data_struct, data_type, params):
    # input
    sent_words = data_struct['sentences']

    # words
    org_sent_words = sent_words['sent_words']
    sent_words = prep_sentences(sent_words, data_type, params) # deal with singletons
    wordsIDs = _elem2idx(sent_words, params['mappings']['word_map'])

    all_sentences = []

    # nner: Using subwords:
    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    for xx, sid in enumerate(data_struct['input']):
        # input
        sentence_data = data_struct['input'][sid]

        # document id
        fid = sid.split(':')[0]

        # words to ids
        word_ids = wordsIDs[xx]
        words = org_sent_words[xx]

        # entity
        readable_e, idxs, ents, toks2, etypes2ids, entities, sw_sentence, sw_sentence_arg, sub_to_word, subwords, valid_starts, tagsIDs, terms = entity2network(
            sentence_data, words, params, tokenizer)
        
        
        # return
        sentence_vector = OrderedDict()
        sentence_vector['fid'] = fid
        sentence_vector['ents'] = ents # not used
        sentence_vector['e_ids'] = idxs  # not used
        sentence_vector['etypes2'] = etypes2ids # not used 
        sentence_vector['toks2'] = toks2 # not used
        
        sentence_vector['word_ids'] = word_ids
        sentence_vector['words'] = words
        sentence_vector['offsets'] = sentence_data['offsets']
        
        sentence_vector['tags'] = tagsIDs
        
        sentence_vector['raw_words'] = sentence_data['words']

        sentence_vector['entities'] = entities
        sentence_vector['sw_sentence'] = sw_sentence
        sentence_vector['sw_sentence_arg'] = sw_sentence_arg
        sentence_vector['terms'] = terms
        
        sentence_vector['sub_to_word'] = sub_to_word
        sentence_vector['subwords'] = subwords
        sentence_vector['valid_starts'] = valid_starts
        
        sentence_vector['tags_terms'] = sentence_data['tags_terms']
        sentence_vector['tags_arg'] = sentence_data['tags_arg']
        sentence_vector['tags_arg_terms'] = sentence_data['tags_arg_terms']
        sentence_vector['terms_ev'] = sentence_data['terms_ev']
        sentence_vector['readable_ents'] = sentence_data['readable_ents']       
        
                
        all_sentences.append(sentence_vector)

    return all_sentences


def torch_data_2_network(cdata2network, params, args, do_get_nn_data):
    """ Convert object-type data to torch.tensor type data, aim to use with Pytorch
    """
    etypes = [data['etypes2'] for data in cdata2network]

    # nner
    entitiess = [data['entities'] for data in cdata2network]
    sw_sentences = [data['sw_sentence'] for data in cdata2network]
    sw_sentences_arg =  [data['sw_sentence_arg'] for data in cdata2network]
    termss = [data['terms'] for data in cdata2network]
    
    terms_evs = [data['terms_ev'] for data in cdata2network]
    readable_entss = [data['readable_ents'] for data in cdata2network]
    
    valid_startss = [data['valid_starts'] for data in cdata2network]

    fids = [data['fid'] for data in cdata2network]
    wordss = [data['words'] for data in cdata2network]
    offsetss = [data['offsets'] for data in cdata2network]
    sub_to_words = [data['sub_to_word'] for data in cdata2network]
    subwords = [data['subwords'] for data in cdata2network]

    tokenizer = BertTokenizer.from_pretrained(
        params['bert_model'], do_lower_case=False
    )

    # User-defined data
    if not params["predict"]:
        id_tag_mapping = params["mappings"]["nn_mapping"]["id_tag_mapping"]

        mlb = MultiLabelBinarizer()
        mlb.fit([sorted(id_tag_mapping)[1:]])  # [1:] skip label O

        params["mappings"]["nn_mapping"]["mlb"] = mlb
        params["mappings"]["nn_mapping"]["num_labels"] = len(mlb.classes_)

        params["max_span_width"] = max(params["max_entity_width"], params["max_trigger_width"])

        params["mappings"]["nn_mapping"]["num_triggers"] = len(params["mappings"]["nn_mapping"]["trigger_labels"])
        params["mappings"]["nn_mapping"]["num_entities"] = params["mappings"]["nn_mapping"]["num_labels"] - \
                                                           params["mappings"]["nn_mapping"]["num_triggers"]

    if do_get_nn_data:
        nn_data = get_nn_data(fids, entitiess, termss, valid_startss, 
                              sw_sentences, sw_sentences_arg, 
                              terms_evs, readable_entss,
                              tokenizer, params, args)

        
        # add entity/protien positions --> {entity/mention label : positions = [star,end]}
        entityTs = []
        for each_terms in termss:
            each_entityT = {}
            for _entity_posi in each_terms:
                
                # if _entity is only starting with "T"
                _entity = each_terms[_entity_posi][0]
                if _entity.startswith("T") and not _entity.startswith("TR"):
                    each_entityT[_entity] = [_entity_posi[0], _entity_posi[1]]

            entityTs.append(each_entityT)
            
            
        
        return {'nn_data': nn_data, 'etypes': etypes, 'fids': fids, 'words': wordss, 'offsets': offsetss,
                'sub_to_words': sub_to_words, 'subwords': subwords, 'entities': entitiess, 'termss': termss, 'entityTs':entityTs, 'terms_evs': terms_evs}
