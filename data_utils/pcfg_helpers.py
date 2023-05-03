from vocabulary import WordVocabulary
from datasets import Dataset as HFDataset
import sequence
import pickle
import random
from collections import Counter

def copy(sequence):
    return (sequence)

def reverse(sequence):
    return (sequence[::-1])

def shift(sequence):
    return (sequence[1:] + [sequence[0]])

def echo(sequence):
    return (sequence + [sequence[-1]])

# def echo_first(sequence):
#     return ([sequence[0]] + sequence)

def swap_first_last(sequence):
    return([sequence[-1]] + sequence[1:-1] + [sequence[0]])

def repeat(sequence):
    return(sequence + sequence)

# BINARY

def append(sequence1, sequence2):
    return (sequence1 + sequence2)

def prepend(sequence1, sequence2):
    return (sequence2 + sequence1)


def remove_first(sequence1, sequence2):
    return (sequence2)

def remove_second(sequence1, sequence2):
    return(sequence1)

def interleave_first(sequence1, sequence2):
    ret_list = []
    idx1 = 0
    idx2 = 0
    while idx1 < len(sequence1) or idx2 < len(sequence2):
        if idx1 < len(sequence1):
            ret_list.append(sequence1[idx1])
            idx1 += 1
        if idx2 < len(sequence2):
            ret_list.append(sequence2[idx2])
            idx2 += 1
    return ret_list

def interleave_second(sequence1, sequence2):
    return interleave_first(sequence2, sequence1)

# singleton lists and use append to create lists of size 2.

fns = {'copy': copy, 'reverse': reverse, 'shift': shift, 'echo': echo, 
       'swap_first_last': swap_first_last, 
       'repeat': repeat, 
       'append': append,
        'prepend': prepend,
        'remove_first': remove_first,
        'remove_second': remove_second,
        'interleave_first': interleave_first,
        'interleave_second': interleave_second}


def is_type(token):
    if token in ['append', 'prepend', 'remove_first', 'remove_second', 'interleave_first', 'interleave_second']:
        return 2
    elif token in ["copy", "reverse", "shift", "echo", "repeat", 'swap_first_last']:
        return 1
    else:
        return 0

def convert_to_tuple(seq):
    def helper(idx, curr_chunk):
        if seq[idx] == '(':
            return helper(idx+1, curr_chunk)
        elif seq[idx]  == ')':
            return (curr_chunk.lstrip(), idx)
        elif seq[idx] == ',':
            if curr_chunk == '':
                return helper(idx+1, curr_chunk)
            else:
                return (curr_chunk.lstrip(), idx)
        else:
            ctype = is_type(seq[idx])
            if ctype == 1:
                p1 = seq[idx]
                p2, advanced = helper(idx+1, '')
                if seq[advanced] != ')':
                    raise Exception
                return (p1, p2), advanced+1
            elif ctype == 2:
                p1 = seq[idx]
                p2, advanced = helper(idx + 1, '')
                p3, advanced2 = helper(advanced+1, '')
                return (p1, (p2, p3)), advanced2+1
            else:
                return helper(idx+1, '{} {}'.format(curr_chunk, seq[idx]))
    out, _= helper(0, '')
    return out

def get_singletons(input_strs):
    input_str_singletons = [convert_all_lists_to_singleton(inp) for inp in input_strs]
    gold_parses_singleton = [place_brackets(inp) for inp in input_str_singletons]
    return input_str_singletons, gold_parses_singleton

def convert_tuple_to_str(ctuple):
    def helper(t):
        if type(t) == str:
            return t
        else:
            all_c = ' '.join([helper(c) for c in t])
            return '( {} )'.format(all_c)
    return helper(ctuple).split(' ')

def place_brackets(seq):
    if type(seq) is str:
        seq = seq.split()
    seq.append("END")
    queue = []
    new_seq = []
    for token in seq:
        if token in ['append', 'prepend', 'remove_first', 'remove_second', 'interleave_first', 'interleave_second']:
            new_seq.append(token)
            new_seq.append("(")
            queue.append(["two-place", 0])
        elif token in ["copy", "reverse", "shift", "echo", "repeat", 'swap_first_last']:
            new_seq.append(token)
            new_seq.append("(")
            queue.append(["one-place", 0])
        elif token == "," or token == "END":
            while len(queue) > 0:
                if queue[-1][0] == "one-place":
                    _ = queue.pop()
                    new_seq.append(")")
                elif queue[-1][0] == "two-place" and queue[-1][1] == 0:
                    queue[-1][1] = 1
                    break
                elif queue[-1][0] == "two-place" and queue[-1][1] == 1:
                    new_seq.append(")")
                    _ = queue.pop()
            if token == "," : new_seq.append(token)
        else:
            new_seq.append(token)
    assert new_seq.count("(") == new_seq.count(")"), "Number of opening and closing brackets do not match."
    return convert_to_tuple(new_seq)

def eval_fn(o):
    if type(o) == str:
        return o.split(' ')
    fn = fns[o[0]]
    num_args = is_type(o[0])
    if num_args == 1:
        return fn(eval_fn(o[1]))
    else:
        return fn(eval_fn(o[1][0]), eval_fn(o[1][1]))

def get_brackets(parse):
    p_set = set()
    def get_brackets_helpers(t, st):
        if type(t) == str:
            return len(t.split(' '))
        else:
            l1_len = get_brackets_helpers(t[0], st)
            l2_len = get_brackets_helpers(t[1], st + l1_len)
            p_set.add((st, st + l1_len + l2_len-1))
            return l1_len + l2_len    
    get_brackets_helpers(parse, 0)
    return p_set

def get_split_indices(parse):
    p_set = {}
    def get_brackets_helpers(t, st):
        if type(t) == str:
            return len(t.split(' '))
        else:
            l1_len = get_brackets_helpers(t[0], st)
            l2_len = get_brackets_helpers(t[1], st + l1_len)
            #p_set.add((st, st + l1_len + l2_len-1))
            p_set[(st, st+l1_len+l2_len-1)] = st + l1_len-1
            return l1_len + l2_len    
    get_brackets_helpers(parse, 0)
    return p_set

def read_data_pcfg(base_folder, splits, use_singleton, use_no_commas, tree_transform=False):
    in_sentences = []
    out_sentences = []
    parses = []
    index_map = {split: [] for split in splits}

    for split in splits:
        fn_name = '{}/{}'.format(base_folder, split)
        if use_singleton:
            fn_name += '_singleton'
        with open('{}.src'.format(fn_name), 'r') as f1, open('{}.tgt'.format(fn_name), 'r') as f2:
            all_inps = [line.strip() for line in f1]
            all_tgts = [line.strip() for line in f2]
            for inp, tgt in zip(all_inps, all_tgts):
                index_map[split].append(len(in_sentences))
                in_sentences.append(inp)
                out_sentences.append(tgt)
                if tree_transform:
                    parses.append(get_split_indices(tree_transformation(place_brackets(inp))))
                else:
                    parses.append(get_split_indices(place_brackets(inp)))
        assert len(in_sentences) == len(out_sentences)
    if use_no_commas:
        in_sentences = [inp.replace(' ,', '') for inp in in_sentences]
    return in_sentences, parses, out_sentences, index_map

def get_examples(split):
    in_sentences, out_sentences, _ = read_data_pcfg([split])
    return in_sentences, out_sentences



def build_datasets_pcfg(use_singleton=False, use_no_commas=False, base_folder='m-pcfgset', tree_transform=False): 
    splits = ['train', 'val', 'gen']
    in_sentences, gold_parses, out_sentences, index_map = read_data_pcfg(base_folder, splits, use_singleton, use_no_commas, tree_transform)
    print('num examples: {}'.format(len(in_sentences)))
    in_vocab = WordVocabulary(in_sentences)
    out_vocab = WordVocabulary(out_sentences)

    def get_subset(elem_list, idx_list):
        return [elem_list[idx] for idx in idx_list]


    dataset = {}
    for split in splits:
        in_subset = get_subset(in_sentences, index_map[split])
        out_subset = get_subset(out_sentences, index_map[split])
        in_subset_tokenized = [in_vocab(s) for s in in_subset]
        out_subset_tokenized = [out_vocab(s) for s in out_subset]
        in_lens = [len(s) for s in in_subset_tokenized]
        out_lens = [len(s) for s in out_subset_tokenized]
        data = {'in': in_subset_tokenized, 
                'idxs': index_map[split],
                'out': out_subset_tokenized, 
                'in_len': in_lens,
                'out_len': out_lens}
        dataset_curr = HFDatasetNoTypes.from_dict(data)
        dataset_curr.set_ds(in_vocab, out_vocab)
        dataset[split] = dataset_curr
    return dataset, in_vocab, out_vocab, in_sentences, gold_parses


class HFDatasetNoTypes(HFDataset):
    def set_ds(self, in_vocab, out_vocab, is_pretrained=False, **kwargs):
        self.in_vocab = in_vocab
        self.out_vocab = out_vocab
        self.is_pretrained = is_pretrained

    def start_test(self):
        if self.is_pretrained:
            # in_vocab and out_vocab are tokenizers
            return sequence.TextSequenceTestState(self.in_vocab.decode,
                                          self.out_vocab.decode)
        else:
            return sequence.TextSequenceTestState(lambda x: " ".join(self.in_vocab(x)),
                                          lambda x: " ".join(self.out_vocab(x)))


def is_leaf_fn_pcfg(word_list, st, en):
    ### check if the words from st to en are a "leaf"
    if st == en:
        return True

    def is_arg(idx, word):
        if idx not in [0, en - st]:
            return len(word) <= 3 and ',' not in word
        else:
            return len(word) <= 3
    return all(is_arg(idx, word) for idx, word in enumerate(word_list[st:en+1]))


def is_invalid_fn_pcfg(word_list, st, k, en):
     if is_type(word_list[st]) == 2:
        return k != st and word_list[k] != ',' and word_list[k+1] != ','
     elif ',' in ' '.join(word_list[st: en+1]):
         #print('here') 
         #print(word_list, word_list[k], word_list[k+1])   
         return word_list[k] != ',' and word_list[k+1] != ','
     elif is_type(word_list[st]) == 1 and is_leaf_fn_pcfg(word_list, st+1, en):
         return k != st
     else:
         return False

def convert_all_lists_to_singleton(input_str):
    bracketed_expr = place_brackets(input_str)
    def helper_fn(inp):
        if type(inp) == str:
            if inp not in fns:
                words = inp.split(' ')
                random.shuffle(words)
                if len(words) > 1:
                    return '{} {}'.format(words[0], words[1])
                else:
                    return words[0]
                #return words[0]
            else:
                return inp
        else:
            c1 = helper_fn(inp[0])
            c2 = helper_fn(inp[1])
            return (c1, c2)
    def helper_fn2(inp, flag):
        if type(inp) == str:
            return inp
        else:
            c1 = helper_fn2(inp[0], False)
            if type(inp[0]) == str and is_type(inp[0]) == 2:
                c2 = helper_fn2(inp[1], True)
            else:
                c2 = helper_fn2(inp[1], False)
            if flag:
                return '{} , {}'.format(c1, c2)
            else:
                return '{} {}'.format(c1, c2)
    return helper_fn2(helper_fn(bracketed_expr), False)

def get_all_constituents(bracketed_inp):
    all_constituents = []
    def helper_fn(inp):
        if type(inp) == str:
            inp_processed = inp.replace(' ,', '')
            inp_processed = inp.replace(',', '')
            all_constituents.append(inp_processed.strip())
            return inp_processed.strip()
        else:
            all_tokens = [helper_fn(c) for c in inp]
            all_tokens = [tok for tok in all_tokens if len(tok) > 0]
            curr_constituent =' '.join(all_tokens)
            all_constituents.append(curr_constituent.strip())
            return curr_constituent.strip()
    helper_fn(bracketed_inp)
    def is_leaf(constituent):
        words = constituent.split(' ')
        def is_arg(word):
            return len(word) <= 3
        return len(words) == 1 or all(is_arg(word) for word in words)
    return [c for c in all_constituents if not is_leaf(c)]

# converts (p (A B)) into ((p A) B)
def tree_transformation(bracketed_inp):
    def helper_fn(inp):
        if type(inp) == str:
            return inp
        else:
            c1, c2 = inp
            if is_type(c1) == 2:
                c21, c22 = helper_fn(c2[0]), helper_fn(c2[1])
                return ((c1, c21), c22)
            else:
                return (helper_fn(c1), helper_fn(c2))
    return helper_fn(bracketed_inp)


def convert_to_linearized_rep_pcfg(parse):
    def helper_fn(inp):
        if type(inp) == str:
            return inp
        else:
            c1, c2 = inp
            s1 = helper_fn(c1)
            s2 = helper_fn(c2)
            return '( {} {} )'.format(s1, s2)
    return helper_fn(parse)

def convert_to_singleton_all(remove_commas=False, base_folder='m-pcfgset'):
    def helper_fn(split):
        with open('{}/{}.src'.format(base_folder, split)) as f1:
            all_inps = [line.strip() for line in f1]    
            inps, parses = get_singletons(all_inps)
            labels = [' '.join(eval_fn(bracket)) for bracket in parses]
            if remove_commas:
                inps = [inp.replace(' ,', '') for inp in inps]
            return inps, parses, labels
    
    def write_to_file(inps, labels, file_name):
        with open('{}.src'.format(file_name), 'w') as f:
            for inp in inps:
                f.write(inp + '\n')

        with open('{}.tgt'.format(file_name), 'w') as f:
            for label in labels:
                f.write(label + '\n')


    for split in ['train', 'val', 'gen']:
        inp, parses, labels = helper_fn(split)
        if remove_commas:
            write_to_file(inp, labels, '{}/{}_singleton_no_commas'.format(base_folder, split))
            with open('{}/pcfg_{}_singleton_no_commas.pickle'.format(base_folder, split), 'wb') as writer:
                pickle.dump((inp, parses), writer)
        else:
            write_to_file(inp, labels, '{}/{}_singleton'.format(base_folder, split))
            with open('{}/pcfg_{}_singleton.pickle'.format(base_folder, split), 'wb') as writer:
                pickle.dump((inp, parses), writer)

def get_parsing_accuracy_pcfg(parses, gold_parses, take_best=True):
    gc = [get_all_constituents(parse) for parse in gold_parses]
    gc_t = [get_all_constituents(tree_transformation(parse)) for parse in gold_parses]
    pc = [get_all_constituents(parse) for parse in parses]

    def get_score(l1, l2):
        counter_l1 = Counter(l1)
        counter_l2 = Counter(l2)
        score = 0.0
        for p in counter_l2:
            if p in counter_l1 and counter_l1[p] == counter_l2[p]:
                score += 1
        return score 

    # to restrict to noun phrases, precision means how many of the NP constituents discovered are actual NPs
    # recall means how many of the NP constituents in the gold are also constituents in our model. 
    precision = 0.0
    recall = 0.0
    recall_sum = 0.0
    for idx, pred in enumerate(pc):
        p1 = get_score(gc[idx], pred)    
        p2 = get_score(gc_t[idx], pred)
        if take_best:
            if p1 > p2:
                precision += p1
                recall += get_score(pred, gc[idx])
                recall_sum += len(gc[idx])
            else:
                precision += p2
                recall += get_score(pred, gc_t[idx])
                recall_sum += len(gc_t[idx])
        else:
            precision += p1
            recall += get_score(pred, gc[idx])
            recall_sum += len(gc[idx])

    precision /= (1.0 * sum(len(b) for b in pc))
    recall /= (1.0 * recall_sum) 
    return {'precision': precision, 'recall': recall, 
            'f1': 2.0 * precision * recall / (precision + recall+1e-10)}
