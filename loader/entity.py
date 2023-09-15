"""Process entity."""

from collections import defaultdict
from collections import OrderedDict
import numpy as np
import re
import math

def process_etypes(entities0):
    entities1 = OrderedDict()
    etypes_ = []
    arguments = []
    
    argtypes = []
    
    for pmid in entities0:
        entities = entities0[pmid]
        etypes = entities['counted_types']
        for type in etypes:
            if type not in etypes_:
                etypes_.append(type)

        events = entities['event']
        for Eid, each_E in events.items():
            for _argtp_each in each_E['arg_type']:
                if _argtp_each not in argtypes:
                    argtypes.append(_argtp_each)
        
        
        for idT in entities['data']:
            argument = entities['data'][idT]['text']
            arguments.append(argument)

    entities1['pmids'] = entities0
    entities1['types'] = etypes_
    entities1['arg_types'] = argtypes
    entities1['arguments'] = arguments

    return entities1


def process_tags(entities1):
    typesT = entities1['types']

    tags = []

    tags2types = OrderedDict()
    tags2types['O'] = 'O'
    for type in typesT:
        btag = 'B-' + type
        itag = 'I-' + type
        tags.append(btag)
        tags.append(itag)
        tags2types[btag] = type
        tags2types[itag] = type

    arg_tags = []
    arg_tags2arg_types = OrderedDict()
    arg_tags2arg_types['O'] = 'O'
    
    
    for arg_tp in entities1['arg_types']:
        
        m = re.search(r'\d+$', arg_tp)
        if m is not None:
            star_, end_ = m.regs[0]
            arg_tp = arg_tp[:star_]
        
        if arg_tp not in arg_tags:
            arg_tags.append(arg_tp)
            arg_tags2arg_types[arg_tp] = arg_tp
        # btag = 'B-' + arg_tp
        # itag = 'I-' + arg_tp
        # arg_tags.append(btag)
        # arg_tags.append(itag)
        # arg_tags2arg_types[btag] = arg_tp
        # arg_tags2arg_types[itag] = arg_tp        
        
    
    
    tags0 = OrderedDict()
    tags0['types'] = typesT
    tags0['typesT'] = typesT
    tags0['tags'] = tags
    tags0['tags2types'] = tags2types

    
    entities1['arg_types'] = arg_tags
    tags0['arg_types'] = entities1['arg_types']
    tags0['arg_tags'] = arg_tags
    tags0['arg_tags2types'] = arg_tags2arg_types
    
    
    return tags0


def assign_label(offsets, terms, terms_ev):
    """
    Assign BIO label to each word of the sentence.
    """
    terms_sentence = []

    if len(terms) == 0:
        lst = [['O'] * len(offsets)]
        return lst, lst, lst, lst, terms_sentence

    max_level = max([item[-1] for item in terms])
    lst = []
    # nner
    lst_term = []
    
    tag_ev = OrderedDict()
    tag_ev_term = OrderedDict()
    
    exist_trig = {}
    
    for _ in range(max_level):
        label = [['O'] * len(offsets)]
        label_term = [['O'] * len(offsets)]
        lst.extend(label)
        # nner
        lst_term.extend(label_term)

        # event arguements
        label_arg = [['O'] * len(offsets)]
        label_arg_term = [['O'] * len(offsets)]
        
    for level in range(max_level):
        terms_level = [item for item in terms if item[-1] == level + 1]
        for i, offset in enumerate(offsets):
            for term in terms_level:
                t_start, t_end = int(term[2]), int(term[3])
                if offset[0] == t_start and offset[1] <= t_end:
                    lst[level][i] = 'B-' + term[1]
                    # nner
                    lst_term[level][i] = 'B-' + term[0]
                    terms_sentence.append(term)

                    if term[0] not in exist_trig:
                        exist_trig[term[0]] = []
                        exist_trig[term[0]].append(i)
                        
                                            
                elif offset[0] > t_start and offset[1] <= t_end:
                    lst[level][i] = 'I-' + term[1]
                    # nner
                    lst_term[level][i] = 'I-' + term[0]
                    
                    exist_trig[term[0]].append(i)
            
                
    
    exist_tg_id = []
    for i in exist_trig.keys():
        exist_tg_id.append(i)
    
    event_dic = {}
    
    terms_ev_sentence = {}
    # initialize the arguement tags and tag-terms             
    for ev_term_each in terms_ev:
        # ev_term_each[2] : trigger id
        
        event_dic[ev_term_each[0]] = ev_term_each
                
        if ev_term_each[2] in exist_tg_id:
            tag_ev[ev_term_each[0]] = {'trig':ev_term_each[2],
                                       'tag_arg_term':[['O'] * len(offsets)]} # all 'O' = 'O'*len(offset)
            tag_ev_term[ev_term_each[0]] = {'trig':ev_term_each[2],
                                            'tag_arg':[['O'] * len(offsets)]} # all 'O' = 'O'*len(offset)
            
            terms_ev_sentence[ev_term_each[0]] = ev_term_each
    # label the arguement tags and tag-terms by triggers        
    for _ev_id in tag_ev.keys():
        ev_info = tag_ev[_ev_id]
        tri_posi = exist_trig[ev_info['trig']]
        
        for i in tri_posi:
            tag_ev[_ev_id]['tag_arg_term'][0][i] = ev_info['trig']
            tag_ev_term[_ev_id]['tag_arg'][0][i] = event_dic[_ev_id][1]
    
        
    # label the arguement tags and tag-terms by arguements
    for _ev_id in tag_ev.keys():
        # argument entity
            #number of arguments
        num_arg = math.floor(len(event_dic[_ev_id][3]))    
        arguements = event_dic[_ev_id][3:]
        for idx_arg in range(num_arg):
            
            arg_type = arguements[0][idx_arg]
            arg_entity = arguements[1][idx_arg]
            
            while arg_entity.startswith('E'):
                if arg_entity not in tag_ev:
                    break
                else:
                    temp_ev_id  = arg_entity
                    arg_entity = tag_ev[temp_ev_id]['trig']
            
            if arg_entity not in exist_trig:
                continue
            else:
                ent_posi = exist_trig[arg_entity]
                # label the arguements
                
                for i in ent_posi:
                    tag_ev[_ev_id]['tag_arg_term'][0][i] = arg_entity
                    m = re.search(r'\d+$', arg_type)
                    if m is not None:
                        star_, end_ = m.regs[0]
                        arg_type = arg_type[:star_]
                    tag_ev_term[_ev_id]['tag_arg'][0][i] = arg_type
                        
    # nner
    # return lst, terms_sentence
    return lst, lst_term, tag_ev, tag_ev_term,  terms_sentence, terms_ev_sentence


def argsort(arr):
    return sorted(range(len(arr)), key=arr.__getitem__)


def count_nest_level(arr, _):
    """
    Calculate nest level of each term and
    get the max nest level.
    term: id, type, start, end, text, nest_level
    """
    # Nest level of flat entities and non-entities is 1
    max_level = 1
    if len(arr) == 0:
        return max_level, arr

    sorted_ids = argsort([[int(e[2]), int(e[3])] for e in arr])

    first_item = arr[sorted_ids[0]]
    first_item.append(max_level)

    levels = defaultdict(list, {max_level: [first_item]})
    for idx in map(lambda p: sorted_ids[p], range(1, len(arr))):
        level = 1
        while level <= max_level:
            if int(arr[idx][2]) >= int(levels[level][-1][3]):
                break
            level += 1

        arr[idx].append(level)
        levels[level].append(arr[idx])
        max_level = max(max_level, level)

    return max_level, arr


def spliter(line, _len=len):
    """
        Credits to https://stackoverflow.com/users/1235039/aquavitae
        Return a list of words and their indexes in a string.
    """
    words = line.split(' ')
    index = line.index
    offsets = []
    append = offsets.append
    running_offset = 0
    for word in words:
        word_offset = index(word, running_offset)
        word_len = _len(word)
        running_offset = word_offset + word_len
        append((word, word_offset, running_offset))
    return offsets


def process_entities(entities1, sentences1, params, dirpath):
    entities0 = entities1['pmids']

    input0 = OrderedDict()

    sentences0 = sentences1['doc_data']
    levels = []

    for pmid in entities0:
        entities = entities0[pmid]
        sentences = sentences0[pmid]

        terms = entities['terms']

        ev_terms = entities['termsE']
        
        nest_level, terms = count_nest_level(terms, params)
        levels.append(nest_level)

        # if nest_level >1:
        #     print(11)
        
        
        abst_text = '\n'.join([sent['sentence'] for sent in sentences])
        spans = []
        init_char = 0
        next_char = 0
        for char in abst_text:
            if char != '\n':
                next_char += 1
            else:
                spans.append((init_char, next_char))
                next_char += 1
                init_char = next_char
        spans.append((init_char, next_char))

        for xx, sentence in enumerate(sentences):
            offsets = sentence['offsets']

            tags, tags_terms, tag_ev, tag_ev_term, terms_sentence, terms_ev_sentence = assign_label(offsets, terms, 
                                                                                                    ev_terms)

            sentence['tags'] = tags
            sentence['tags_terms'] = tags_terms
            sentence['terms'] = terms_sentence


            sentence['tags_arg'] = tag_ev_term
            sentence['tags_arg_terms'] = tag_ev
            sentence['terms_ev'] = terms_ev_sentence
            
            
            
            eids = []
            for t1 in terms_sentence:
                eid = t1[0]
                eids.append(eid)
            sentence['eids'] = eids
            readable_ents = OrderedDict()
            for eid in eids:
                if eid in entities['data']:
                    readable_ents[eid] = entities['data'][eid]

            span = spans[xx]

            for x, id_ in enumerate(eids):  # for every entity if it belongs to sentence span
                ent = readable_ents[id_]
                b = int(ent['pos1'])
                e = int(ent['pos2'])
                if (span[0] <= b <= span[1]) and (span[0] <= e <= span[1]):
                    b2 = b - span[0]
                    e2 = e - span[0]

                    ent['offs2'] = [b2, e2]
                else:
                    print("SKIP ENTITY: " + str(b) + " --- " + str(e))

            sentence['readable_ents'] = readable_ents

            tokens = spliter(
                sentence['sentence'])  # we have the tokens of the sentence and their corresponding offsets

            for eid in eids:
                if "offs2" not in readable_ents[eid]:
                    print(readable_ents[eid])
                    continue
                offs = readable_ents[eid]['offs2']
                start = offs[0]
                end = offs[1]
                toks = []
                for tok_id, (tok, start0, end0) in enumerate(tokens):  # of the word token
                    if (start0, end0) == (start, end):
                        toks.append(tok_id)
                    elif start0 == start and end0 < end:
                        toks.append(tok_id)
                    elif start0 > start and end0 < end:
                        toks.append(tok_id)
                    elif start0 > start and end0 == end:
                        toks.append(tok_id)

                readable_ents[eid]['toks'] = toks

    max_nest_level = max(levels)
    max_nest_level += 1

    for pmid in sentences1['doc_data']:
        in_sentences = sentences1['doc_data'][pmid]
        out_sentences = []
        label_count = len(in_sentences[0]['tags'])
        pad_level = max_nest_level - label_count

        for xx, sentence in enumerate(in_sentences):
            tags = sentence['tags']
            pad_label = [['O'] * len(tags[0])]
            tags.extend(pad_label * pad_level)

            tags_terms = sentence['tags_terms']
            pad_label = [['O'] * len(tags_terms[0])]
            tags_terms.extend(pad_label * pad_level)

            out_sentences.append(sentence)

        input0[pmid] = out_sentences

    return input0


def extract_entities(sw_sentence, tag2id_mapping, id2tag_mapping, nn_mapping):
    # For several edge cases
    max_depth = max(len(tags) for _, tags, _ in sw_sentence)

    entities = defaultdict(list)
    terms = defaultdict(list)

    
    
    tokens = [token for token, *_ in sw_sentence]

    num_tokens = len(tokens)

    begin_indices = np.arange(num_tokens)
    end_indices = begin_indices + 1

    token_indices = np.column_stack((begin_indices, end_indices))

    try:
        tags = np.asarray(
            [
                [tag2id_mapping[tag] for tag in tags + ["O"] * max_depth][
                :max_depth
                ]
                for _, tags, tags_terms in sw_sentence
            ]
        ).T
    except KeyError as err:
        tags = np.asarray(
            [
                [tag2id_mapping[tag] if tag in tag2id_mapping else tag2id_mapping["O"] for tag in
                 tags + ["O"] * max_depth][
                :max_depth
                ]
                for _, tags, tags_terms in sw_sentence
            ]
        ).T
        print(err)

    tags_terms = np.asarray(
        [
            [tag_term for tag_term in tags_terms + ["O"] * max_depth][
            :max_depth
            ]
            for _, _, tags_terms in sw_sentence
        ]
    ).T

    for idx, tag in enumerate(tags):
        tag_term = tags_terms[idx]
        bo_indices = np.where(tag % 2 == 0)[0]
        i_indices = np.where(tag % 2 == 1)[0]

        num_rows, num_cols = tag.shape[0], bo_indices.shape[0]

        # Build merging matrix
        merging_matrix = np.full((num_rows, num_cols), 0, dtype=np.int)
        merging_matrix[bo_indices, np.arange(num_cols)] = 1

        # Fill I tags
        sub_indices = [tag[:i] for i in i_indices]
        counts = np.asarray(
            [np.where(i % 2 == 0)[0].shape[0] for i in sub_indices],
            dtype=np.int,
        )

        merging_matrix[i_indices, counts - 1] = 1

        # Get all the start indices for each token
        token_start_indices = np.zeros(merging_matrix.shape, dtype=np.int)
        token_start_indices[
            np.argmax(merging_matrix, axis=0), np.arange(num_cols)
        ] = 1
        token_start_indices = np.matmul(
            token_indices[:, 0], token_start_indices
        )

        # Get all the end indices for each token
        flipped_merging_matrix = np.flipud(merging_matrix)
        token_end_indices = np.zeros(merging_matrix.shape, dtype=np.int)
        token_end_indices[
            num_rows - 1 - np.argmax(flipped_merging_matrix, axis=0),
            np.arange(num_cols),
        ] = 1
        token_end_indices = np.matmul(
            token_indices[:, 1], token_end_indices
        ) - 1

        for begin_pos, end_pos in zip(
                token_start_indices.flatten(), token_end_indices.flatten()
        ):
            tag_id = tag[begin_pos]
            term_name = tag_term[begin_pos]
            if tag_id > 0:  # Tag O is not an entity
                tag_name = re.sub("^[BI]-", "", id2tag_mapping[tag_id])
                term_name = re.sub("^[BI]-", "", term_name)

                entities[(begin_pos, end_pos)].append(nn_mapping['tag_id_mapping'][tag_name])
                terms[(begin_pos, end_pos)].append(term_name)

    return entities, terms, sw_sentence


def convert_to_sub_words(word_tokens, tags, tags_terms, tags_arg, tags_arg_terms, tokenizer=None):
    subword_pos = 0
    subword_offset_mapping = {}
    subwords = []
    sw_sentence = []
    
    sw_sentence_arg = {}  
    # initialize the sw_sentence_arg
    for ev_id in tags_arg.keys():
        sw_sentence_arg[ev_id] = []
    
    
    valid_starts = [0]

    for token_idx, token in enumerate(word_tokens):
        if tokenizer:
            subtokens = tokenizer.tokenize(token)
            if subtokens:
                
                sw_sentence.append(subtokens[:1] + [tags[token_idx],tags_terms[token_idx]])
                subword_offset_mapping[subword_pos] = token_idx
                subword_pos += 1
                subwords.append(subtokens[:1][0])

                for ev_id in tags_arg.keys():
                    sw_sentence_arg[ev_id].append(subtokens[:1] + 
                                                [tags_arg[ev_id]['tag_arg'][0][token_idx],
                                                tags_arg_terms[ev_id]['tag_arg_term'][0][token_idx]])
                
                
                labels = [re.sub("^B-", "I-", label) for label in tags[token_idx]]
                ids = [re.sub("^B-", "I-", _id) for _id in tags_terms[token_idx]]

                for subtoken in subtokens[1:]:
                    sw_sentence.append([subtoken] + [labels, ids])
                    
                    subword_offset_mapping[subword_pos] = token_idx
                    subword_pos += 1
                    subwords.append(subtoken)
                    
                    for ev_id in tags_arg.keys():
                        sw_sentence_arg[ev_id].append([subtoken] + 
                                                    [tags_arg[ev_id]['tag_arg'][0][token_idx],
                                                    tags_arg_terms[ev_id]['tag_arg_term'][0][token_idx]])
                    
                    

            valid_starts.append(len(subwords))
        else:
            sw_sentence.append([token] + [tags[token_idx], tags_terms[token_idx]])
            subword_offset_mapping[token_idx] = token_idx
    return sw_sentence, sw_sentence_arg, subword_offset_mapping, subwords, valid_starts
